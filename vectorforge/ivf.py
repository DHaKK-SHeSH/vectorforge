"""IVF (inverted file) index.  <-- YOU write train, add and search.

Idea: don't compare a query with ALL n base vectors. Instead:
  - train:  run k-means once to get `nlist` centroids (coarse "cells").
  - add:    put every base vector into the list of its nearest centroid.
            lists[j] = ids of the base vectors whose nearest centroid is j.
  - search: for each query, find its `nprobe` nearest centroids and compare it
            exactly with the vectors in those lists only.

nprobe is the speed/recall knob:
  nprobe = 1      -> look in one cell: fastest, can miss neighbours that sit
                     just across a cell border
  nprobe = nlist  -> look everywhere: exactly the same answer as FlatIndex

Goal: make tests/test_ivf.py pass. Squared L2 distance only (like flat.py's 'l2').
"""

import numpy as np

from .kmeans import assign, kmeans


class IVFIndex:
    def __init__(self, nlist: int = 100, nprobe: int = 1, n_iter: int = 20, seed: int = 0):
        self.nlist = nlist
        self.nprobe = nprobe
        self.n_iter = n_iter
        self.seed = seed
        self.centroids = None  # (nlist, d) float32, set by train()
        self.lists = None      # list of nlist int64 arrays of base-vector ids, set by add()
        self.xb = None         # (n, d) float32 base vectors, set by add()

    @property
    def is_trained(self) -> bool:
        return self.centroids is not None

    def train(self, xt: np.ndarray) -> None:
        """Learn the nlist centroids with k-means on the training vectors xt.

        Use a separate training set when you have one (SIFT ships siftsmall_learn),
        so the cells don't depend on what you later add.
        """
        raise NotImplementedError("TODO: run your kmeans and keep the centroids")

    def add(self, xb: np.ndarray) -> None:
        """Store xb and build the inverted lists.

        Hints:
        - assign() from kmeans.py gives each base vector's nearest centroid.
        - For each cell j: np.flatnonzero(labels == j) gives the ids in that cell.
          (Faster, loop-free alternative once it works: np.argsort(labels) groups
          the ids by cell, and np.bincount gives where each group starts.)
        - Raise an error if add() is called before train().
        """
        raise NotImplementedError("TODO: build the inverted lists")

    def search(self, xq: np.ndarray, k: int):
        """Return (distances, ids), both (nq, k), squared L2, nearest first.

        For each query:
          1. find its nprobe nearest centroids
          2. gather the candidate ids from those lists (np.concatenate)
          3. compute exact squared distances to just those candidates
          4. keep the k nearest, sorted

        If a query has fewer than k candidates, fill the missing slots with
        id -1 and distance np.inf (that's what Faiss does too).

        A Python loop over QUERIES is fine here (each query has a different
        candidate set). Never loop over individual vectors.
        Step 1 can be done for all queries at once: it's a FlatIndex search
        over the centroids.
        """
        raise NotImplementedError("TODO: search only the nprobe nearest cells")
