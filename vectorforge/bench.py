"""Benchmark harness: recall@k, queries per second and index memory.

Every index in this project (flat, IVF, PQ, HNSW, Faiss) must expose
    index.add(xb)  and  index.search(xq, k) -> (distances, ids)
so they can all be measured by this same function.

Run from the project root:
    python -m vectorforge.bench
"""

import time
from typing import Dict

import numpy as np

from .metrics import recall_at_k


def index_nbytes(index) -> int:
    """Rough memory of an index: sum of the numpy arrays it holds."""
    total = 0
    for value in vars(index).values():
        if isinstance(value, np.ndarray):
            total += value.nbytes
        elif isinstance(value, (list, tuple)):
            total += sum(v.nbytes for v in value if isinstance(v, np.ndarray))
    return total


def benchmark(index, xb: np.ndarray, xq: np.ndarray, gt: np.ndarray, k: int = 10, repeats: int = 3) -> Dict[str, float]:
    t0 = time.perf_counter()
    index.add(xb)
    build_s = time.perf_counter() - t0

    best = float("inf")
    ids = None
    for _ in range(repeats):  # best of several runs: less noise from the OS
        t0 = time.perf_counter()
        _, ids = index.search(xq, k)
        best = min(best, time.perf_counter() - t0)

    return {
        "recall@k": recall_at_k(ids, gt, k),
        "qps": len(xq) / best,
        "build_s": build_s,
        "memory_mb": index_nbytes(index) / 1e6,
    }


def main() -> None:
    from .flat import FlatIndex
    from .io import load_sift

    xb, xq, gt = load_sift()
    print(f"base {xb.shape}, queries {xq.shape}, ground truth {gt.shape}")
    result = benchmark(FlatIndex("l2"), xb, xq, gt, k=10)
    for key, value in result.items():
        print(f"{key:>10}: {value:.4f}")


if __name__ == "__main__":
    main()
