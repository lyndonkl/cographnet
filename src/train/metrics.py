from typing import Dict, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(labels: List[int], predictions: List[int]) -> Dict[str, float]:
    """Calculate classification metrics."""
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='macro'),
        'recall': recall_score(labels, predictions, average='macro'),
        'f1': f1_score(labels, predictions, average='macro')
    } 