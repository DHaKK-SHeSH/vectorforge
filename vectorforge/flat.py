"""Exact (brute-force) k-nearest-neighbour search.  <-- YOU write this file.

This is the baseline every other index is measured against, so it must be
exactly right. Goal for week 1: make tests/test_flat.py pass.

The idea to use (this week's Monday topic):
    ||q - x||^2 = ||q||^2 - 2 q.x + ||x||^2
so all query-to-base distances are ONE matrix multiply (queries @ base.T)
plus two cheap vector terms. No Python loop over vectors.
"""

import numpy as np


class FlatIndex:
    def __init__(self, metric: str = "l2"):
        if metric not in ("l2", "ip"):
            raise ValueError("metric must be 'l2' or 'ip'")
        self.metric = metric
        self.xb = None  # the stored base vectors, shape (n, d), float32

    def add(self, xb: np.ndarray) -> None:
        """Store the base vectors.
        """
        self.xb = np.asarray(xb,dtype=np.float32)
        self.xb_norms = (self.xb**2).sum(axis=1)
        
        

    def search(self, xq: np.ndarray, k: int):
        """Return (distances, ids), both of shape (nq, k), nearest first.

        metric 'l2': distances are SQUARED L2 distances, sorted ascending.
        metric 'ip': distances are inner products, sorted DESCENDING (bigger = closer).

        Hints:
        - build the (nq, n) distance matrix with one matrix multiply
        - np.argpartition finds the k smallest in O(n); then sort only those k
        - floating-point error can make tiny squared distances slightly negative;
          think about whether to clip them
        """
        xq = np.asarray(xq,dtype=np.float32)
        if self.metric == "ip":
            D = xq @ self.xb.T
        else:
            q_norms = (xq**2).sum(axis=1)
            D = q_norms[:, None] - 2 * (xq @ self.xb.T) + self.xb_norms
            D = np.maximum(D, 0)  
        
        if self.metric == "ip":
            D_sorted = -D
        else:
            D_sorted = D

        part = np.argpartition(D_sorted, k-1, axis=1)[:,:k]
        dist = np.take_along_axis(D, part, axis=1)
        order = np.argsort(np.take_along_axis(D_sorted, part, axis=1), axis=1)
        ids = np.take_along_axis(part, order, axis=1)                 # ids sorted by distance
        distances = np.take_along_axis(dist, order, axis=1)
        return distances,ids
