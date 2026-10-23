"""Product quantisation (PQ).  <-- YOU write train, encode, decode, add and search.

Idea: store a tiny CODE instead of the full vector.

  1. Cut every d-dimensional vector into m pieces ("sub-vectors") of d/m numbers.
     SIFT: 128 numbers -> m = 8 pieces of 16 numbers.
  2. For each piece position j, run k-means on the training data's j-th pieces
     with ksub = 2**nbits centroids (256 by default). That list of 256 piece-
     centroids is "codebook j".
  3. ENCODE a vector: for each piece, store only the NUMBER (0..255) of the
     nearest centroid in that piece's codebook. One byte per piece.
     SIFT: 128 floats x 4 bytes = 512 bytes  ->  8 bytes.  64x smaller.
  4. DECODE: look the numbers up in the codebooks and glue the pieces back.
     You get an APPROXIMATION of the original vector.
  5. SEARCH without decoding (asymmetric distance computation, ADC):
     for a query, precompute a table  T[j, c] = || query piece j - centroid c of codebook j ||^2
     Then the distance to any stored code is just m table lookups added up:
         dist(query, code) = sum_j T[j, code[j]]
     That equals || query - decode(code) ||^2 exactly. A test checks this.

Goal: make tests/test_pq.py pass. Squared L2 only.
"""

import numpy as np

from .kmeans import assign, kmeans


class PQIndex:
    def __init__(self, m: int = 8, nbits: int = 8, n_iter: int = 20, seed: int = 0):
        if not 1 <= nbits <= 8:
            raise ValueError("nbits must be between 1 and 8 (codes are stored as uint8)")
        self.m = m
        self.nbits = nbits
        self.ksub = 2 ** nbits     # centroids per codebook
        self.n_iter = n_iter
        self.seed = seed
        self.dsub = None           # numbers per piece (d // m), set by train()
        self.codebooks = None      # (m, ksub, dsub) float32, set by train()
        self.codes = None          # (n, m) uint8, set by add()

    @property
    def is_trained(self) -> bool:
        return self.codebooks is not None

    def _split(self, x: np.ndarray) -> np.ndarray:
        """(n, d) -> (n, m, dsub): piece j of vector i is x_split[i, j]. (Given.)"""
        return x.reshape(len(x), self.m, self.dsub)

    def train(self, xt: np.ndarray) -> None:
        """Learn the m codebooks.

        Steps:
        - d = xt.shape[1]; raise ValueError if d is not divisible by m.
        - self.dsub = d // self.m
        - pieces = self._split(xt)          -> pieces[:, j, :] is every training vector's j-th piece
        - for each j: run your kmeans on pieces[:, j, :] with k = self.ksub
          (use seed=self.seed + j so each codebook gets a different random start)
        - stack the m centroid arrays into self.codebooks with shape (m, ksub, dsub)
          (np.stack(list_of_arrays) does that)
        """
        xt = np.asarray(xt, dtype=np.float32)
        d = xt.shape[1]
        
        if (d%self.m)!=0:
            raise ValueError(f"d = {d} is not divisible by m = {self.m}")
        
        self.dsub = d//self.m
        pieces = self._split(xt)
        
        books = []
        for j in range(self.m):
            centroids, _, _ = kmeans(pieces[:,j,:], self.ksub, n_iter=self.n_iter,seed = self.seed + j)
            books.append(centroids)
        
        self.codebooks = np.stack(books)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Return codes (n, m) uint8: codes[i, j] = nearest centroid number for piece j of x[i].

        Hint: for each j, assign(pieces[:, j, :], self.codebooks[j]) gives the labels.
        """
        x = np.asarray(x , dtype=np.float32)
        pieces = self._split(x)
        cols = []
        for j in range(self.m):
            labels,_ = assign(pieces[:, j, :], self.codebooks[j])
            cols.append(labels)
        
        return np.stack(cols,axis=1).astype(np.uint8)
        
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Return the approximate vectors (n, d) float32 rebuilt from the codes.

        Hint: self.codebooks[j][codes[:, j]] gives every vector's j-th piece
        (shape (n, dsub)); np.concatenate([...], axis=1) glues the m pieces together.
        """
        parts = []
        for j in range(self.m):
            catalouge = self.codebooks[j]
            numbers = codes[:,j]
            parts.append(catalouge[numbers])
        return np.concatenate(parts,axis=1)

    def add(self, xb: np.ndarray) -> None:
        """Encode the database and keep ONLY the codes (that's the memory saving)."""
        self.codes = self.encode(xb)
        
    def distances(self, xq: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Distance from every query to every code in `codes`. Returns (nq, len(codes))."""
        xq = np.asarray(xq, dtype=np.float32)
        q_pieces = self._split(xq)
        D = np.zeros((len(xq), len(codes)), dtype=np.float32)

        for j in range(self.m):
            qj = q_pieces[:, j, :]
            cat = self.codebooks[j]

            q_norms = (qj**2).sum(axis=1)
            cat_norms = (cat**2).sum(axis=1)
            table = q_norms[:, None] - 2 * (qj @ cat.T) + cat_norms

            D += table[:, codes[:, j]]

        return np.maximum(D, 0)

    def search(self, xq: np.ndarray, k: int):
        """Return (distances, ids), both (nq, k), approximate squared L2, nearest first."""
        D = self.distances(xq, self.codes)

        part = np.argpartition(D, k-1, axis=1)[:, :k]
        dist = np.take_along_axis(D, part, axis=1)
        order = np.argsort(dist, axis=1)
        ids = np.take_along_axis(part, order, axis=1)
        distances = np.take_along_axis(dist, order, axis=1)
        return distances, ids
            
            
