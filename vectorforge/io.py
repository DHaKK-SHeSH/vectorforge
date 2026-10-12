"""Readers for the TEXMEX .fvecs / .ivecs formats (SIFT10K, SIFT1M, ...).

File layout: every vector is stored as a little-endian int32 dimension d,
followed by d values (float32 for .fvecs, int32 for .ivecs). All vectors in
a file have the same d, so the file is an (n, 1 + d) block of 4-byte words.
"""

from pathlib import Path
from typing import Union

import numpy as np

PathLike = Union[str, Path]


def _read_vecs(path: PathLike, dtype: np.dtype) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"{path} is empty")
    d = int(raw[0])
    if d <= 0 or raw.size % (d + 1) != 0:
        raise ValueError(f"{path}: size {raw.size} is not a multiple of d+1={d + 1}")
    block = raw.reshape(-1, d + 1)
    if not np.all(block[:, 0] == d):
        raise ValueError(f"{path}: vectors have inconsistent dimensions")
    return np.ascontiguousarray(block[:, 1:].view(dtype))


def read_fvecs(path: PathLike) -> np.ndarray:
    """Return an (n, d) float32 array."""
    return _read_vecs(path, np.dtype(np.float32))


def read_ivecs(path: PathLike) -> np.ndarray:
    """Return an (n, d) int32 array (ground-truth neighbour ids)."""
    return _read_vecs(path, np.dtype(np.int32))


def write_fvecs(path: PathLike, x: np.ndarray) -> None:
    """Write an (n, d) array as .fvecs. Used by the tests."""
    x = np.asarray(x, dtype=np.float32)
    n, d = x.shape
    out = np.empty((n, d + 1), dtype=np.int32)
    out[:, 0] = d
    out[:, 1:] = x.view(np.int32)
    out.tofile(path)


def load_sift(root: PathLike = "data/siftsmall", prefix: str = "siftsmall"):
    """Load base, query and ground truth for SIFT10K (prefix 'siftsmall') or SIFT1M (prefix 'sift')."""
    root = Path(root)
    base = read_fvecs(root / f"{prefix}_base.fvecs")
    query = read_fvecs(root / f"{prefix}_query.fvecs")
    gt = read_ivecs(root / f"{prefix}_groundtruth.ivecs")
    return base, query, gt
