"""The real done-signal for week 1: exact search reproduces SIFT10K's ground truth.

Skipped automatically until the data is downloaded (see data/README.md), and
always skipped in CI, since the dataset is not committed to git.
"""

from pathlib import Path

import pytest

from vectorforge.flat import FlatIndex
from vectorforge.io import load_sift
from vectorforge.metrics import recall_at_k

DATA = Path(__file__).resolve().parents[1] / "data" / "siftsmall"


@pytest.mark.skipif(not (DATA / "siftsmall_base.fvecs").exists(), reason="SIFT10K not downloaded")
def test_exact_search_reproduces_ground_truth():
    xb, xq, gt = load_sift(DATA, "siftsmall")
    index = FlatIndex("l2")
    index.add(xb)
    _, ids = index.search(xq, k=10)
    # SIFT contains a few exact duplicate vectors, so ties can swap ids; allow a hair below 1.0
    assert recall_at_k(ids, gt, 10) >= 0.999
