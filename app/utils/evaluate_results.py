import json
from pathlib import Path

def calculate_metrics_from_file(file_path: Path) -> dict:
    """
    Calculate precision, recall, and F1-score from a results JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for item in data['results']:
        gt_uris = {tag['uri'] for tag in item['ground_truth']}

        pred_data = item.get('predicted', {})
        if pred_data.get('error'):
            pred_uris = set()
        else:
            pred_uris = {tag['uri'] for tag in pred_data.get('entities', [])}

        tp = len(gt_uris.intersection(pred_uris))
        fp = len(pred_uris - gt_uris)
        fn = len(gt_uris - pred_uris)

        true_positives += tp
        false_positives += fp
        false_negatives += fn

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }