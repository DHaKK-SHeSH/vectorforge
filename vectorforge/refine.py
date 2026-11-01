"""Exact re-ranking on top of any approximate index.

The problem: IVF-PQ is small and fast, but its distances are approximate, so
recall stops at about 0.61 on SIFT10K (0.37 on SIFT1M).

The fix (two stages):
  1. ask the approximate index for a SHORTLIST that is bigger than k:
     k * k_factor candidates (e.g. 10 * 10 = 100)
  2. measure the EXACT distance from the query to each candidate, using the
     original vectors, and keep the k closest

The approximate index only has to get the true neighbours somewhere into the
shortlist, not into the right order. Only k * k_factor full vectors are read per
query, so the full vectors can stay on disk (a memory-mapped .npy file) while only
the small codes sit in RAM.

Example (PQ got the order wrong):
    query [28, 30, 3, 3];  id 0 = [29, 31, 2, 2];  id 2 = [30, 28, 3, 3]
    PQ says id 2 (4.0), then id 0 (6.0).  k = 1, k_factor = 2:
    shortlist [2, 0] -> exact distances: id 2 = 8, id 0 = 4 -> answer id 0 (correct)

Works with any index that has add() and search() (PQ, IVF-PQ, HNSW, ...).
This is what Faiss calls IndexRefineFlat.
"""

import numpy as np


class RefineIndex:
    def __init__(self, base, k_factor: int = 10):
        self.base = base            # the approximate index that makes the shortlist
        self.k_factor = k_factor    # shortlist size = k * k_factor
        self.xb = None              # (n, d) the ORIGINAL vectors, used only for re-ranking

    @property
    def is_trained(self) -> bool:
        return getattr(self.base, "is_trained", True)

    def train(self, xt: np.ndarray) -> None:
        """Train the base index, if it needs training (HNSW doesn't have a train method)."""
        if hasattr(self.base, "train"):
            self.base.train(xt)

    def add(self, xb: np.ndarray) -> None:
        """Add the vectors to the base index AND keep the original vectors for re-ranking.
        A np.memmap (vectors on disk) is kept as it is, not copied into RAM."""
        self.base.add(xb)
        self.xb = xb if isinstance(xb, np.memmap) else np.asarray(xb, dtype=np.float32)

    def search(self, xq: np.ndarray, k: int):
        """Return (distances, ids), both (nq, k), EXACT squared L2, nearest first.
        Pad with -1 / inf if fewer than k candidates were found."""
        xq = np.asarray(xq, dtype=np.float32)

        # 1. the shortlist from the approximate index
        _, shortlist = self.base.search(xq, k * self.k_factor)       # (nq, k * k_factor), -1 = padding

        # 2. empty answer sheets
        nq = len(xq)
        D = np.full((nq, k), np.inf, dtype=np.float32)
        I = np.full((nq, k), -1, dtype=np.int64)

        for qi in range(nq):
            cand = shortlist[qi][shortlist[qi] >= 0]                   # real ids only, no -1 padding
            if len(cand) == 0:
                continue
            # 3. exact distances with the original vectors (only these rows are read from disk)
            vectors = np.asarray(self.xb[cand], dtype=np.float32)
            d = ((vectors - xq[qi]) ** 2).sum(axis=1)
            # 4. keep the k closest, sorted
            kk = min(k, len(cand))
            order = np.argsort(d)[:kk]
            D[qi, :kk] = d[order]
            I[qi, :kk] = cand[order]
        return D, I
