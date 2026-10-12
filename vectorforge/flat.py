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

        Hint: for 'l2' you can precompute ||x||^2 for every base vector here,
        so search() does not redo it for every query batch.
        """
        raise NotImplementedError("TODO: store xb (and precompute what search needs)")

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
        raise NotImplementedError("TODO: exact search")
