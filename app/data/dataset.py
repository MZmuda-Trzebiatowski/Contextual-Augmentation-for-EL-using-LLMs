import json
from pathlib import Path
from typing import Optional, Union

from torch.utils.data import Dataset

from app.ollama.models import ELTagExtend


class EntityLinkingDataset(Dataset):
    """
    PyTorch Dataset for Entity Linking tasks.
    
    Loads JSON files containing corpus text and ground truth entity tags.
    Each item contains the original text and its associated entity annotations.
    
    Args:
        json_path: Path to a JSON file or directory containing JSON files.
        dataset_name: Optional name for the dataset (defaults to filename).
    """
    
    def __init__(
        self, 
        json_path: Union[str, Path], 
        dataset_name: Optional[str] = None
    ) -> None:
        self.json_path = Path(json_path)
        self.dataset_name = dataset_name or self.json_path.stem
        self.data: list[dict] = []
        
        self._load_data()
    
    def _load_data(self) -> None:
        """Load data from JSON file(s)."""
        if self.json_path.is_file():
            self._load_json_file(self.json_path)
        elif self.json_path.is_dir():
            for json_file in sorted(self.json_path.glob("*.json")):
                self._load_json_file(json_file)
        else:
            raise FileNotFoundError(f"Path not found: {self.json_path}")
    
    def _load_json_file(self, file_path: Path) -> None:
        """Load a single JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        
        for idx, item in enumerate(content):
            if not item or "corpus" not in item:
                continue

            ground_truth_tags = []
            for tag in item.get("tags", []):
                if all(k in tag for k in ["text", "beginIndex", "endIndex", "uri"]):
                    ground_truth_tags.append(
                        ELTagExtend(
                            text=tag["text"],
                            uri=tag["uri"],
                            beginIndex=tag["beginIndex"],
                            endIndex=tag["endIndex"]
                        )
                    )
            
            self.data.append({
                "id": f"{file_path.stem}_{idx}",
                "corpus": item["corpus"],
                "ground_truth": ground_truth_tags,
                "source_file": file_path.name
            })
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single item from the dataset.
        
        Returns:
            dict with keys:
                - id: Unique identifier for the sample
                - corpus: The original text
                - ground_truth: List of ELTagExtend objects (ground truth annotations)
                - source_file: Name of the source JSON file
        """
        return self.data[idx]
    
    def get_all_texts(self) -> list[str]:
        """Get all corpus texts for batch processing."""
        return [item["corpus"] for item in self.data]
    
    def get_batch(self, start_idx: int, batch_size: int) -> list[dict]:
        """
        Get a batch of items.
        
        Args:
            start_idx: Starting index
            batch_size: Number of items to retrieve
            
        Returns:
            List of dataset items
        """
        end_idx = min(start_idx + batch_size, len(self.data))
        return [self.data[i] for i in range(start_idx, end_idx)]


def load_all_datasets(jsons_dir: Union[str, Path]) -> dict[str, EntityLinkingDataset]:
    """
    Load all JSON datasets from a directory.
    
    Args:
        jsons_dir: Path to directory containing JSON files
        
    Returns:
        Dictionary mapping dataset names to EntityLinkingDataset instances
    """
    jsons_dir = Path(jsons_dir)
    datasets = {}
    
    for json_file in sorted(jsons_dir.glob("*.json")):
        dataset_name = json_file.stem
        datasets[dataset_name] = EntityLinkingDataset(json_file, dataset_name)
    
    return datasets
