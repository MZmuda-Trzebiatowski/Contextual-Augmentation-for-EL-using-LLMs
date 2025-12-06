import json
import re
from typing import List, Dict, Any

def normalize_uri(uri: str) -> str:
    """
    Normalizes a URI (supporting both Wikipedia and DBpedia patterns) to a 
    lowercased page title for comparison.
    """
    if not uri:
        return ""
        
    uri = uri.lower()
    normalized = uri
    
    if 'wikipedia.org/wiki/' in uri:
        normalized = re.sub(r'https?://[^/]+/wiki/', '', uri)
    
    elif 'dbpedia.org/resource/' in uri:
        normalized = re.sub(r'https?://[^/]+/resource/', '', uri)
        
    normalized = normalized.split('#')[0].split('?')[0]
    
    return normalized

def compute_entity_linking_metrics(data: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Computes Micro-F1, Precision, and Recall for Entity Linking data.
    A match requires an exact span match AND a normalized URI match 
    (using the updated normalize_uri function).
    """
    global_tp = 0
    global_fp = 0
    global_fn = 0

    for doc in data:
        ground_truth = doc.get('ground_truth', [])
        predicted_data = doc.get('predicted', {})
        predicted = predicted_data.get('entities', []) if predicted_data else []

        matched_gt_indices = set()
        matched_pred_indices = set()

        for i_pred, pred_entity in enumerate(predicted):
            for i_gt, gt_entity in enumerate(ground_truth):
                if i_gt in matched_gt_indices:
                    continue

                span_match = (
                    pred_entity.get('beginIndex') == gt_entity.get('beginIndex') and
                    pred_entity.get('endIndex') == gt_entity.get('endIndex')
                )

                uri_match = (
                    normalize_uri(pred_entity.get('uri', '')) == normalize_uri(gt_entity.get('uri', ''))
                )

                if span_match and uri_match:
                    global_tp += 1
                    matched_gt_indices.add(i_gt)
                    matched_pred_indices.add(i_pred)
                    break 

        global_fn += len(ground_truth) - len(matched_gt_indices)

        global_fp += len(predicted) - len(matched_pred_indices)

    total_predicted = global_tp + global_fp
    micro_precision = global_tp / total_predicted if total_predicted > 0 else 0.0

    total_ground_truth = global_tp + global_fn
    micro_recall = global_tp / total_ground_truth if total_ground_truth > 0 else 0.0

    if micro_precision + micro_recall > 0:
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
    else:
        micro_f1 = 0.0

    return {
        "TP": global_tp,
        "FP": global_fp,
        "FN": global_fn,
        "Micro_Precision": micro_precision,
        "Micro_Recall": micro_recall,
        "Micro_F1_Score": micro_f1
    }

def load_json_and_evaluate(file_path: str) -> Dict[str, Any]:
    """
    Loads data from a JSON file, handles nested structures (like 'results'),
    runs the Entity Linking evaluation, and returns the computed metrics.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        metadata = {}
        evaluation_data = []

        if isinstance(data, dict):
            metadata = data.get('metadata', {})
            if 'results' in data and isinstance(data['results'], list):
                evaluation_data = data['results']
            else:
                raise ValueError("JSON dictionary must contain a 'results' key with a list of documents.")
        elif isinstance(data, list):
            evaluation_data = data
        else:
            raise ValueError("JSON file content is neither a list nor a dictionary with a 'results' key.")

        metrics = compute_entity_linking_metrics(evaluation_data)

        return {
            "file_path": file_path,
            "metadata": metadata,
            "metrics": metrics
        }

    except FileNotFoundError:
        return {"error": f"File not found at path: {file_path}"}
    except json.JSONDecodeError:
        return {"error": f"Error decoding JSON from file: {file_path}. Check file format."}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}
    

#file_to_evaluate = 'KORE50_gemma3_4b_results.json'
#file_to_evaluate = 'MSNBCt_gemma3_4b_results.json'
#file_to_evaluate = 'RSS-500_gemma3_4b_results.json'
#file_to_evaluate = 'Reuters-128_gemma3_4b_results.json'
file_to_evaluate = 'evaluation-dataset-task1_gemma3_4b_results.json'
final_results = load_json_and_evaluate(file_to_evaluate)
print(json.dumps(final_results, indent=4))