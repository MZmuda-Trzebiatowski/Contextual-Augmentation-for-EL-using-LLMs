import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.data.dataset import EntityLinkingDataset, load_all_datasets
from app.ollama.llm_service import OllamaGPT, SimpleOllamaGPT
from app.ollama.enhanced_llm_service import EnhancedOllamaGPT
from app.ollama.models import ELTagExtend


def serialize_tag(tag: ELTagExtend) -> dict:
    """Convert ELTagExtend to dictionary for JSON serialization."""
    return {
        "text": tag.text,
        "uri": tag.uri,
        "beginIndex": tag.beginIndex,
        "endIndex": tag.endIndex
    }


def process_dataset(
    dataset: EntityLinkingDataset,
    model: OllamaGPT,
    max_workers: int = 4,
    limit: Optional[int] = None
) -> list[dict]:
    """
    Process a dataset using the OllamaGPT model.
    
    Args:
        dataset: EntityLinkingDataset to process
        model: OllamaGPT model instance
        max_workers: Number of concurrent workers for batch processing
        limit: Optional limit on number of samples to process
        
    Returns:
        List of result dictionaries
    """
    if limit:
        items = [dataset[i] for i in range(min(limit, len(dataset)))]
    else:
        items = [dataset[i] for i in range(len(dataset))]
    
    texts = [item["corpus"] for item in items]
    
    print(f"Processing {len(texts)} samples from {dataset.dataset_name}...")
    
    batch_results = model.run_batch(texts, max_workers=max_workers)
    
    results = []
    for item, batch_result in zip(items, batch_results):
        result = {
            "id": item["id"],
            "corpus": item["corpus"],
            "source_file": item["source_file"],
            "ground_truth": [serialize_tag(tag) for tag in item["ground_truth"]],
            "predicted": {
                "ner_output": batch_result["ner_output"],
                "entities": [serialize_tag(tag) for tag in batch_result["entities"]] if batch_result["entities"] else [],
                "error": batch_result["error"]
            }
        }
        results.append(result)
    
    return results


def save_results(
    results: list[dict],
    dataset_name: str,
    model_name: str,
    results_dir: Path
) -> Path:
    """
    Save processing results to a JSON file.
    
    Args:
        results: List of result dictionaries
        dataset_name: Name of the dataset
        model_name: Name of the model used
        results_dir: Directory to save results
        
    Returns:
        Path to the saved results file
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "metadata": {
            "dataset": dataset_name,
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(results),
            "successful": sum(1 for r in results if r["predicted"]["error"] is None),
            "failed": sum(1 for r in results if r["predicted"]["error"] is not None)
        },
        "results": results
    }
    
    safe_model_name = model_name.replace(":", "_").replace("/", "_")
    output_path = results_dir / f"{dataset_name}_{safe_model_name}_results.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Process entity linking datasets with OllamaGPT"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="gemma3:4b",
        help="Ollama model to use (default: gemma3:4b)"
    )
    parser.add_argument(
        "--dataset", 
        type=str,
        help="Specific dataset to process (e.g., KORE50, MSNBCt)"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Process all datasets in the jsons directory"
    )
    parser.add_argument(
        "--jsons-dir",
        type=str,
        default="data/jsons",
        help="Path to the directory containing JSON datasets"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Path to the directory where results will be saved"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of concurrent workers (default: 4)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples to process per dataset"
    )
    
    args = parser.parse_args()
    
    if not args.dataset and not args.all:
        parser.error("Either --dataset or --all must be specified")
    
    jsons_dir = Path(args.jsons_dir)
    results_dir = Path(args.results_dir)
    
    if not jsons_dir.exists():
        print(f"Error: JSON directory not found: {jsons_dir}")
        sys.exit(1)

    print(f"Initializing model: {args.model}")
    model = EnhancedOllamaGPT(args.model)
    
    if args.all:
        datasets = load_all_datasets(jsons_dir)
        print(f"Found {len(datasets)} datasets: {list(datasets.keys())}")
    else:
        dataset_path = jsons_dir / f"{args.dataset}.json"
        if not dataset_path.exists():
            print(f"Error: Dataset not found: {dataset_path}")
            print(f"Available datasets: {[f.stem for f in jsons_dir.glob('*.json')]}")
            sys.exit(1)
        datasets = {args.dataset: EntityLinkingDataset(dataset_path)}
    
    all_results = {}
    for name, dataset in datasets.items():
        print(f"\n{'='*60}")
        print(f"Processing dataset: {name} ({len(dataset)} samples)")
        print(f"{'='*60}")
        
        results = process_dataset(
            dataset, 
            model, 
            max_workers=args.max_workers,
            limit=args.limit
        )
        
        output_path = save_results(results, name, args.model, results_dir)
        all_results[name] = {
            "output_path": str(output_path),
            "total": len(results),
            "successful": sum(1 for r in results if r["predicted"]["error"] is None),
            "failed": sum(1 for r in results if r["predicted"]["error"] is not None)
        }

    print("Processing Summary")
    for name, stats in all_results.items():
        print(f"  {name}: {stats['successful']}/{stats['total']} successful")
    print(f"\nAll results saved to: {results_dir}/")


if __name__ == "__main__":
    main()
