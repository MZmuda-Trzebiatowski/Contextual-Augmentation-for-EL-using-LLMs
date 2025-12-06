import argparse
import json
import sys
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.data.dataset import EntityLinkingDataset, load_all_datasets
from app.ollama.llm_service import OllamaGPT
from app.ollama.enhanced_llm_service import EnhancedOllamaGPT
from app.ollama.models import ELTagExtend
from app.utils.evaluate_results import calculate_metrics_from_file

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

def append_metrics_to_csv(
    metrics: dict,
    dataset_name: str,
    model_name: str,
    results_dir: Path
):
    """
    Zapisuje metryki do pliku CSV o nazwie: model_name_dataset_name_results.csv
    """
    safe_model_name = model_name.replace(":", "_").replace("/", "_")
    
    # ZMIANA: Nazwa pliku zawiera teraz Model ORAZ Dataset
    csv_filename = f"{safe_model_name}_{dataset_name}_results.csv"
    csv_path = results_dir / csv_filename
    
    file_exists = csv_path.exists()
    
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Jeśli plik nie istnieje, dodaj nagłówek
        if not file_exists:
            writer.writerow(["Timestamp", "Precision", "Recall", "F1"])
        
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1']:.4f}"
        ])
    
    print(f"Metrics appended to: {csv_path}")
    print(f"  -> F1: {metrics['f1']:.4f} | P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f}")

def main():
    parser = argparse.ArgumentParser(
        description="Process entity linking datasets with OllamaGPT"
    )
    parser.add_argument("--model", type=str, default="gemma3:4b", help="Ollama model to use")
    parser.add_argument("--dataset", type=str, help="Specific dataset to process")
    parser.add_argument("--all", action="store_true", help="Process all datasets")
    parser.add_argument("--jsons-dir", type=str, default="data/jsons")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    
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
    else:
        dataset_path = jsons_dir / f"{args.dataset}.json"
        if not dataset_path.exists():
            print(f"Error: Dataset not found: {dataset_path}")
            sys.exit(1)
        datasets = {args.dataset: EntityLinkingDataset(dataset_path)}
    
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

        try:
            metrics = calculate_metrics_from_file(output_path)
            append_metrics_to_csv(metrics, name, args.model, results_dir)
        except Exception as e:
            print(f"Error calculating metrics for {name}: {e}")

    print(f"\nAll processing complete. Check {results_dir} for details.")

if __name__ == "__main__":
    main()