"""Evaluation metrics.  

recall@k (the definition used by ANN-Benchmarks):
    for each query, take the true k nearest neighbours (ground truth) and the
    k ids your index returned; recall is |true ∩ returned| / k.
    Report the mean over all queries.

Example: true top-3 = {4, 7, 9}, returned = [7, 2, 4]  ->  recall@3 = 2/3.
"""

import numpy as np


def recall_at_k(found_ids: np.ndarray, true_ids: np.ndarray, k: int) -> float:
    """found_ids: (nq, >=k) ids returned by an index, nearest first.
    true_ids:  (nq, >=k) ground-truth ids, nearest first.
    Only the first k columns of each are used. Returns a float in [0, 1].
    """
    counter = 0
    recall = 0
    for f, t in zip(found_ids, true_ids):
        match_count = len(set(f[:k]) & set(t[:k]))
        recall += match_count / k
        counter +=1
    
    return recall/counter
