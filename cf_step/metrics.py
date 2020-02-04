# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/metrics.ipynb (unless otherwise specified).

__all__ = ['recall_at_k', 'precision_at_k']

# Cell
from typing import List

# Cell
def recall_at_k(predictions: List[int], targets: List[int], k: int):
    """Computes `Recall@k` from the given predictions and targets sets."""
    predictions_set = set(predictions[:k])
    targets_set = set(targets)
    result = len(targets_set & predictions_set) / float(len(targets_set))
    return result

# Cell
def precision_at_k(predictions: List[int], targets: List[int], k: int):
    """Computes `Precision@k` from the given predictions and targets sets."""
    predictions_set = set(predictions[:k])
    targets_set = set(targets)
    result = len(targets_set & predictions_set) / float(len(predictions_set))
    return result