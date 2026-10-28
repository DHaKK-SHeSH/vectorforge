"""Tests for vectorforge/hnsw.py, in the order you write it:

    pytest tests/test_hnsw.py -k level           # step 1: _random_level
    pytest tests/test_hnsw.py -k search_layer    # step 2: _search_layer
    pytest tests/test_hnsw.py -k graph           # step 3: _insert
    pytest tests/test_hnsw.py -k query           # step 4: search
"""

from pathlib import Path

import numpy as np
import pytest

from vectorforge.flat import FlatIndex
from vectorforge.hnsw import HNSWIndex
from vectorforge.metrics import recall_at_k

DATA = Path(__file__).resolve().parents[1] / "data" / "siftsmall"


# ---------------------------------------------------------------- step 1: levels
def test_level_is_a_non_negative_int():
    index = HNSWIndex(M=16, seed=0)
    for _ in range(200):
        level = index._random_level()
        assert isinstance(level, int) and level >= 0


def test_level_distribution_thins_out_by_M_per_layer():
    """With mL = 1/ln(M): about 1 in M vectors reaches layer 1, 1 in M^2 reaches layer 2."""
    index = HNSWIndex(M=16, seed=0)
    levels = np.array([index._random_level() for _ in range(40000)])
    assert abs((levels >= 1).mean() - 1 / 16) < 0.01
    assert abs((levels >= 2).mean() - 1 / 256) < 0.002


# ---------------------------------------------------------------- step 2: one layer
def line_index():
    """20 points on a line (x = 0, 1, ..., 19), each linked to the 2 points on either side.
    Only layer 0, built by hand, so _search_layer can be tested before _insert exists."""
    index = HNSWIndex(M=4)
    index.xb = np.arange(20, dtype=np.float32)[:, None]
    index.graph = [{i: [j for j in (i - 2, i - 1, i + 1, i + 2) if 0 <= j < 20] for i in range(20)}]
    return index


def test_search_layer_walks_to_the_nearest():
    index = line_index()
    found = index._search_layer(np.array([13.2], dtype=np.float32), [0], 1, 0)
    assert len(found) == 1
    d, i = found[0]
    assert i == 13 and d == pytest.approx(0.04, abs=1e-4)


def test_search_layer_returns_the_ef_closest_sorted():
    index = line_index()
    found = index._search_layer(np.array([13.2], dtype=np.float32), [0], 5, 0)
    assert [i for _, i in found] == [13, 14, 12, 15, 11]
    np.testing.assert_allclose([d for d, _ in found], [0.04, 0.64, 1.44, 3.24, 4.84], atol=1e-4)


def test_search_layer_never_returns_more_than_ef():
    index = line_index()
    for ef in (1, 3, 7):
        found = index._search_layer(np.array([4.6], dtype=np.float32), [19], ef, 0)
        assert len(found) == ef
        dists = [d for d, _ in found]
        assert dists == sorted(dists)


# ---------------------------------------------------------------- step 3: building the graph
@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    xb = rng.standard_normal((1000, 16)).astype(np.float32)
    xq = rng.standard_normal((30, 16)).astype(np.float32)
    flat = FlatIndex("l2")
    flat.add(xb)
    _, gt = flat.search(xq, 10)
    return xb, xq, gt


@pytest.fixture(scope="module")
def built(data):
    xb, _, _ = data
    index = HNSWIndex(M=8, ef_construction=64, ef_search=64, seed=1)
    index.add(xb)
    return index


def test_graph_first_vector_becomes_the_entry_point():
    index = HNSWIndex(M=8, seed=0)
    index.add(np.zeros((1, 4), dtype=np.float32))
    assert index.entry_point == 0
    assert index.max_level == index.levels[0]


def test_graph_every_vector_is_on_layer_0(built, data):
    xb, _, _ = data
    assert sorted(built.graph[0]) == list(range(len(xb)))


def test_graph_layers_are_nested(built):
    """A vector on layer l is on every layer below it, and on exactly the layers 0..its level."""
    for l, layer in enumerate(built.graph):
        for node in layer:
            assert built.levels[node] >= l
    for node, level in enumerate(built.levels):
        for l in range(level + 1):
            assert node in built.graph[l]


def test_graph_entry_point_is_on_the_top_layer(built):
    assert built.max_level == len(built.graph) - 1
    assert built.levels[built.entry_point] == built.max_level


def test_graph_neighbour_lists_are_valid(built):
    for l, layer in enumerate(built.graph):
        limit = built.M0 if l == 0 else built.M
        for node, nbrs in layer.items():
            assert len(nbrs) <= limit
            assert node not in nbrs
            assert len(set(nbrs)) == len(nbrs)
            assert all(n in layer for n in nbrs)       # neighbours live on the same layer


def test_graph_links_point_to_close_vectors(built, data):
    xb, _, _ = data
    rng = np.random.default_rng(0)
    linked = np.mean([((xb[n] - xb[i]) ** 2).sum() for i, nbrs in built.graph[0].items() for n in nbrs])
    a, b = rng.integers(len(xb), size=(2, 3000))
    random_pairs = ((xb[a] - xb[b]) ** 2).sum(axis=1).mean()
    assert linked < 0.75 * random_pairs


# ---------------------------------------------------------------- step 4: search
def test_query_single_vector():
    index = HNSWIndex(M=8, seed=0)
    index.add(np.array([[1.0, 2.0]], dtype=np.float32))
    d, i = index.search(np.array([[1.0, 3.0]], dtype=np.float32), 1)
    assert i[0, 0] == 0 and d[0, 0] == pytest.approx(1.0)


def test_query_sorted_and_padded():
    index = HNSWIndex(M=8, seed=0)
    index.add(np.arange(10, dtype=np.float32).reshape(5, 2))
    d, i = index.search(np.zeros((2, 2), dtype=np.float32), 8)
    assert d.shape == (2, 8) and i.shape == (2, 8)
    assert np.all(i[:, 5:] == -1) and np.all(np.isinf(d[:, 5:]))
    assert np.all(np.diff(d[:, :5], axis=1) >= 0)


def test_query_distances_are_exact(built, data):
    xb, xq, _ = data
    d, ids = built.search(xq, 10)
    for q in range(len(xq)):
        true = ((xb[ids[q]] - xq[q]) ** 2).sum(axis=1)
        np.testing.assert_allclose(d[q], true, rtol=1e-4, atol=1e-3)


def test_query_recall_on_random_data(built, data):
    _, xq, gt = data
    built.ef_search = 64
    _, ids = built.search(xq, 10)
    assert recall_at_k(ids, gt, 10) >= 0.9


def test_query_bigger_ef_search_finds_more(built, data):
    _, xq, gt = data
    built.ef_search = 10
    _, ids_small = built.search(xq, 10)
    built.ef_search = 200
    _, ids_big = built.search(xq, 10)
    built.ef_search = 64
    assert recall_at_k(ids_big, gt, 10) >= recall_at_k(ids_small, gt, 10)
    assert recall_at_k(ids_big, gt, 10) >= 0.97


@pytest.mark.skipif(not (DATA / "siftsmall_base.fvecs").exists(), reason="SIFT10K not downloaded")
def test_query_sift_recall():
    """2,000 SIFT vectors (a few seconds to build); ground truth from exact search."""
    from vectorforge.io import load_sift
    xb, xq, _ = load_sift(DATA, "siftsmall")
    xb = xb[:2000]
    flat = FlatIndex("l2")
    flat.add(xb)
    _, gt = flat.search(xq, 10)
    index = HNSWIndex(M=16, ef_construction=100, ef_search=64)
    index.add(xb)
    _, ids = index.search(xq, 10)
    assert recall_at_k(ids, gt, 10) >= 0.95
