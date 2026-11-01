"""Tests for vectorforge/refine.py (v1.1 extra A: exact re-ranking).

    pytest tests/test_refine.py
"""

from pathlib import Path

import numpy as np
import pytest

from vectorforge.flat import FlatIndex
from vectorforge.hnsw import HNSWIndex
from vectorforge.ivfpq import IVFPQIndex
from vectorforge.metrics import recall_at_k
from vectorforge.pq import PQIndex
from vectorforge.refine import RefineIndex

DATA = Path(__file__).resolve().parents[1] / "data" / "siftsmall"


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    xt = rng.standard_normal((2000, 16)).astype(np.float32)
    xb = rng.standard_normal((800, 16)).astype(np.float32)
    xq = rng.standard_normal((25, 16)).astype(np.float32)
    flat = FlatIndex("l2")
    flat.add(xb)
    _, gt = flat.search(xq, 10)
    return xt, xb, xq, gt


@pytest.fixture(scope="module")
def refined(data):
    xt, xb, _, _ = data
    index = RefineIndex(PQIndex(m=4, nbits=4), k_factor=10)
    index.train(xt)
    index.add(xb)
    return index


def test_distances_are_exact(refined, data):
    """After re-ranking, distances are the TRUE distances to the original vectors, not PQ's."""
    _, xb, xq, _ = data
    d, ids = refined.search(xq, 10)
    for q in range(len(xq)):
        true = ((xb[ids[q]] - xq[q]) ** 2).sum(axis=1)
        np.testing.assert_allclose(d[q], true, rtol=1e-4, atol=1e-3)


def test_sorted_and_shaped(refined, data):
    _, _, xq, _ = data
    d, ids = refined.search(xq, 10)
    assert d.shape == (25, 10) and ids.shape == (25, 10)
    assert np.all(np.diff(d, axis=1) >= -1e-5)


def test_reranking_beats_the_base_index(refined, data):
    _, _, xq, gt = data
    _, base_ids = refined.base.search(xq, 10)
    _, ids = refined.search(xq, 10)
    assert recall_at_k(ids, gt, 10) > recall_at_k(base_ids, gt, 10) + 0.1


def test_k_factor_1_keeps_the_same_ids(refined, data):
    """With no extra candidates, re-ranking can only re-order the base index's answer."""
    _, _, xq, _ = data
    refined.k_factor = 1
    _, ids = refined.search(xq, 10)
    _, base_ids = refined.base.search(xq, 10)
    refined.k_factor = 10
    for q in range(len(xq)):
        assert set(ids[q]) == set(base_ids[q])


def test_checking_everything_is_exact_search(data):
    """If the shortlist holds every vector, re-ranking IS exact search."""
    xt, xb, xq, gt = data
    index = RefineIndex(PQIndex(m=4, nbits=4), k_factor=len(xb) // 10)
    index.train(xt)
    index.add(xb)
    _, ids = index.search(xq, 10)
    np.testing.assert_array_equal(ids, gt)


def test_padding_when_the_base_finds_too_few(data):
    xt, xb, xq, _ = data
    index = RefineIndex(IVFPQIndex(nlist=4, nprobe=4, m=4, nbits=4), k_factor=5)
    index.train(xt)
    index.add(xb[:6])                    # only 6 vectors stored, but k = 10
    d, ids = index.search(xq[:2], 10)
    assert np.all(ids[:, 6:] == -1) and np.all(np.isinf(d[:, 6:]))
    assert np.all(ids[:, :6] >= 0)


def test_works_with_a_graph_index_that_has_no_train(data):
    xt, xb, xq, gt = data
    index = RefineIndex(HNSWIndex(M=6, ef_construction=40, ef_search=10), k_factor=4)
    index.train(xt)                      # HNSW has no train(): this must simply do nothing
    index.add(xb)
    _, ids = index.search(xq, 10)
    assert recall_at_k(ids, gt, 10) >= 0.8


def test_full_vectors_can_stay_on_disk(data, tmp_path):
    """The full vectors are only read for the shortlist, so they can live in a memory-mapped file."""
    xt, xb, xq, gt = data
    np.save(tmp_path / "xb.npy", xb)
    on_disk = np.load(tmp_path / "xb.npy", mmap_mode="r")
    index = RefineIndex(PQIndex(m=4, nbits=4), k_factor=10)
    index.train(xt)
    index.add(on_disk)
    _, ids = index.search(xq, 10)
    assert recall_at_k(ids, gt, 10) >= 0.85


@pytest.mark.skipif(not (DATA / "siftsmall_base.fvecs").exists(), reason="SIFT10K not downloaded")
def test_sift_ivfpq_recall_jumps():
    from vectorforge.io import load_sift, read_fvecs
    xb, xq, gt = load_sift(DATA, "siftsmall")
    xt = read_fvecs(DATA / "siftsmall_learn.fvecs")
    index = RefineIndex(IVFPQIndex(nlist=100, nprobe=8, m=8), k_factor=10)
    index.train(xt)
    index.add(xb)
    _, ids = index.search(xq, 10)
    assert recall_at_k(ids, gt, 10) >= 0.9        # IVF-PQ alone: about 0.61
