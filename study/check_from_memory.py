"""Checks your from-memory exact search against your FlatIndex.

1. In study/from_memory.py, write (without looking at flat.py):

       def search(xb, xq, k):
           ...
           return distances, ids        # both (nq, k), squared L2, nearest first

2. Run from the project root:

       python study/check_from_memory.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from from_memory import search  # noqa: E402

from vectorforge.flat import FlatIndex  # noqa: E402


def check(name, xb, xq, k):
    flat = FlatIndex("l2")
    flat.add(xb)
    d_true, i_true = flat.search(xq, k)
    d, i = search(xb, xq, k)
    d, i = np.asarray(d), np.asarray(i)
    ok_shape = d.shape == (len(xq), k) and i.shape == (len(xq), k)
    ok_ids = ok_shape and np.array_equal(i, i_true)
    ok_d = ok_shape and np.allclose(d, d_true, rtol=1e-4, atol=1e-2)
    ok_sorted = ok_shape and bool(np.all(np.diff(d, axis=1) >= -1e-4))
    print(f"{name:<22} shape {'OK' if ok_shape else 'WRONG ' + str(d.shape)}   "
          f"ids {'OK' if ok_ids else 'WRONG'}   distances {'OK' if ok_d else 'WRONG'}   "
          f"sorted {'OK' if ok_sorted else 'WRONG'}")
    return ok_ids and ok_d and ok_sorted


rng = np.random.default_rng(0)
results = [
    check("tiny (5 vectors, k=2)", np.arange(10, dtype=np.float32).reshape(5, 2), np.array([[0.0, 0.0], [9.0, 9.0]], dtype=np.float32), 2),
    check("random (2000 x 32)", rng.standard_normal((2000, 32)).astype(np.float32), rng.standard_normal((50, 32)).astype(np.float32), 10),
]
data = Path(__file__).resolve().parents[1] / "data" / "siftsmall"
if (data / "siftsmall_base.fvecs").exists():
    from vectorforge.io import load_sift
    xb, xq, _ = load_sift(data, "siftsmall")
    results.append(check("SIFT10K", xb, xq, 10))
print("\nAll correct!" if all(results) else "\nNot yet: look at the WRONG columns.")
