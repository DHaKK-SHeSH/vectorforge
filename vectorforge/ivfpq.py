"""IVF-PQ: IVF cells + PQ codes.  <-- YOU write train, add and search.

It combines the two indexes you already built:

  - IVF decides WHERE to look: k-means cells, and a query only searches its
    nprobe nearest cells (your ivf.py).
  - PQ decides HOW vectors are stored: m bytes per vector instead of 512
    (your pq.py). Search inside a cell uses the lookup tables.

The residual trick (by_residual=True, what Faiss does by default):
  Instead of encoding the vector itself, encode its OFFSET from the centre of
  its cell:  residual = x - centroids[cell].
  Offsets are small and look alike in every cell, so the 256 catalogue entries
  are spent on fine detail instead of on "which part of space is this in"
  (the cell already tells you that). Same bytes, better recall.

  Then at search time, inside cell c, compare the query's offset from that same
  centre with the stored offsets:
      ||q - (centre + offset)||^2  ==  ||(q - centre) - offset||^2

What the index stores (no full vectors anywhere):
  self.centroids    (nlist, d)        the cell centres
  self.pq           a PQIndex         its codebooks (trained on residuals)
  self.lists        nlist id arrays   which base vectors are in each cell
  self.list_codes   nlist (len, m)    their PQ codes, in the same order as lists

Goal: make tests/test_ivfpq.py pass. Squared L2 only.
"""

import numpy as np

from .flat import FlatIndex
from .kmeans import assign, kmeans
from .pq import PQIndex


class IVFPQIndex:
    def __init__(self, nlist: int = 100, nprobe: int = 1, m: int = 8, nbits: int = 8,
                 by_residual: bool = True, n_iter: int = 20, seed: int = 0):
        self.nlist = nlist
        self.nprobe = nprobe
        self.by_residual = by_residual
        self.n_iter = n_iter
        self.seed = seed
        self.pq = PQIndex(m=m, nbits=nbits, n_iter=n_iter, seed=seed)   # your PQ does the encoding
        self.centroids = None    # (nlist, d) float32, set by train()
        self.lists = None        # list of nlist int64 arrays of base-vector ids, set by add()
        self.list_codes = None   # list of nlist (len, m) uint8 arrays, set by add()

    @property
    def is_trained(self) -> bool:
        return self.centroids is not None and self.pq.is_trained

    def train(self, xt: np.ndarray) -> None:
        """Learn the cells, then the PQ codebooks.

        1. k-means on xt -> self.centroids   (exactly your IVF train)
        2. if self.by_residual: replace xt by each vector's offset from its cell centre
           (assign() gives each vector's cell; self.centroids[labels] gives its centre)
        3. self.pq.train(...) on what you have now
        """
        xt = np.asarray(xt, dtype=np.float32)
        self.centroids, _, _ = kmeans(xt, self.nlist, n_iter=self.n_iter, seed=self.seed)
        
        if self.by_residual:
          labels, _ = assign(xt,self.centroids)
          xt = xt - self.centroids[labels]
          
        self.pq.train(xt)

    def add(self, xb: np.ndarray) -> None:
        """Put every base vector in its cell, and store only its PQ code.

        1. raise an error if not trained (like your IVF add)
        2. labels = each vector's cell (assign)
        3. if self.by_residual: replace xb by the offsets (same as in train)
        4. codes = self.pq.encode(...)          -> (n, m) uint8
        5. self.lists      = ids in each cell (exactly your IVF add)
           self.list_codes = the codes of those ids, one array per cell (codes[ids])
        """
        if not self.is_trained:
          raise RuntimeError("call train() first")
        
        labels, _ = assign(xb,self.centroids)
        
        if self.by_residual:
          xb = xb - self.centroids[labels]
          
        codes = self.pq.encode(xb)
        self.lists = [np.flatnonzero(labels==c) for c in range(self.nlist)]
        
        self.list_codes = [codes[ids] for ids in self.lists]
           

    def search(self, xq: np.ndarray, k: int):
        """Return (distances, ids), both (nq, k), approximate squared L2, nearest first.

        Same shape as your IVF search:
          - find each query's nprobe nearest cells (FlatIndex over the centroids)
          - D = inf, I = -1 (padding when there are fewer than k candidates)
          - loop over queries; for each probed cell c:
                q = the query's offset from centre c (or the query itself if not by_residual)
                d = self.pq.distances(q[None, :], self.list_codes[c])[0]
                collect d and self.lists[c]
          - concatenate, keep the k smallest, sorted (like your IVF search)
        """
        xq = np.asarray(xq, dtype=np.float32)


        finder = FlatIndex("l2")
        finder.add(self.centroids)
        _, cells = finder.search(xq, min(self.nprobe, self.nlist))

        nq = len(xq)
        D = np.full((nq, k), np.inf, dtype=np.float32)
        I = np.full((nq, k), -1, dtype=np.int64)

        for qi in range(nq):
            all_ids, all_d = [], []


            for c in cells[qi]:
                if len(self.lists[c]) == 0:
                    continue
                if self.by_residual:
                    q = xq[qi] - self.centroids[c]                             
                else:
                    q = xq[qi]
                d = self.pq.distances(q[None, :], self.list_codes[c])[0]
                all_d.append(d)
                all_ids.append(self.lists[c])                        

            if not all_ids:
                continue

            cand = np.concatenate(all_ids)
            d = np.concatenate(all_d)
            kk = min(k, len(cand))
            part = np.argpartition(d, kk - 1)[:kk]
            order = part[np.argsort(d[part])]
            D[qi, :kk] = d[order]
            I[qi, :kk] = cand[order]

        return D, I
        
        
