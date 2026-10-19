from pathlib import Path

import numpy as np
import pytest

from vectorforge.flat import FlatIndex
from vectorforge.ivf import IVFIndex
from vectorforge.metrics import recall_at_k

DATA = Path(__file__).resolve().parents[1] / "data" / "siftsmall"


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    xt = rng.standard_normal((2000, 16)).astype(np.float32)   # training set
    xb = rng.standard_normal((1500, 16)).astype(np.float32)   # base
    xq = rng.standard_normal((30, 16)).astype(np.float32)     # queries
    flat = FlatIndex("l2")
    flat.add(xb)
    d_true, i_true = flat.search(xq, 10)
    return xt, xb, xq, d_true, i_true


def make_index(xt, xb, nlist=20, nprobe=1):
    index = IVFIndex(nlist=nlist, nprobe=nprobe)
    index.train(xt)
    index.add(xb)
    return index


def test_add_before_train_raises(data):
    _, xb, *_ = data
    with pytest.raises(Exception):
        IVFIndex(nlist=4).add(xb)


def test_every_vector_in_exactly_one_list(data):
    xt, xb, *_ = data
    index = make_index(xt, xb)
    assert index.centroids.shape == (20, 16)
    assert len(index.lists) == 20
    all_ids = np.sort(np.concatenate(index.lists))
    np.testing.assert_array_equal(all_ids, np.arange(len(xb)))


def test_vectors_sit_in_their_nearest_cell(data):
    xt, xb, *_ = data
    index = make_index(xt, xb)
    for j, ids in enumerate(index.lists):
        if len(ids):
            d = ((xb[ids][:, None, :] - index.centroids[None]) ** 2).sum(-1)
            assert np.all(d.argmin(axis=1) == j)


def test_probing_all_cells_equals_exact_search(data):
    xt, xb, xq, d_true, i_true = data
    index = make_index(xt, xb, nlist=20, nprobe=20)
    d, i = index.search(xq, 10)
    assert d.shape == (30, 10) and i.shape == (30, 10)
    np.testing.assert_array_equal(i, i_true)
    np.testing.assert_allclose(d, d_true, rtol=1e-4, atol=1e-3)


def test_recall_never_drops_when_probing_more(data):
    xt, xb, xq, _, i_true = data
    index = make_index(xt, xb, nlist=20)
    recalls = []
    for nprobe in (1, 2, 4, 8, 20):
        index.nprobe = nprobe
        _, i = index.search(xq, 10)
        recalls.append(recall_at_k(i, i_true, 10))
    assert all(b >= a for a, b in zip(recalls, recalls[1:])), recalls
    assert recalls[-1] == pytest.approx(1.0)
    assert recalls[0] < 1.0, "nprobe=1 should miss some neighbours on random data"


def test_results_sorted_and_padded(data):
    xt, xb, xq, *_ = data
    index = make_index(xt, xb, nlist=20, nprobe=1)
    d, i = index.search(xq, 500)  # more than one cell holds: forces padding
    valid = np.isfinite(d)
    assert np.all(i[~valid] == -1)
    for row in d:
        r = row[np.isfinite(row)]
        assert np.all(np.diff(r) >= 0)


@pytest.mark.skipif(not (DATA / "siftsmall_base.fvecs").exists(), reason="SIFT10K not downloaded")
def test_sift_full_probe_is_exact():
    from vectorforge.io import load_sift, read_fvecs
    xb, xq, gt = load_sift(DATA, "siftsmall")
    xt = read_fvecs(DATA / "siftsmall_learn.fvecs")
    index = make_index(xt, xb, nlist=100, nprobe=100)
    _, i = index.search(xq, 10)
    assert recall_at_k(i, gt, 10) >= 0.999
