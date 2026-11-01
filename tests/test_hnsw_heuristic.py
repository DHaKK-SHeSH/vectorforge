"""v1.1 extra B: HNSW's neighbour-selection heuristic (HNSWIndex(heuristic=True), the default)."""

from pathlib import Path

import numpy as np
import pytest

from vectorforge.flat import FlatIndex
from vectorforge.hnsw import HNSWIndex
from vectorforge.metrics import recall_at_k
from vectorforge.persist import load, save

DATA = Path(__file__).resolve().parents[1] / "data" / "siftsmall"


def picker(heuristic):
    """An index whose only vectors are the candidates below, to test _select_neighbours directly."""
    index = HNSWIndex(M=2, heuristic=heuristic)
    index.xb = np.array([[1.0, 0.0],     # id 0: east
                         [1.1, 0.1],     # id 1: east, right next to id 0
                         [1.1, -0.1],    # id 2: east, right next to id 0
                         [-2.0, 0.0]],   # id 3: west, farther away
                        dtype=np.float32)
    node = np.zeros(2, dtype=np.float32)                     # the new vector, at the origin
    d = ((index.xb - node) ** 2).sum(axis=1)
    candidates = sorted((float(d[i]), i) for i in range(4))  # closest first: 0, 1, 2, 3
    return index, candidates


def test_plain_selection_takes_the_closest():
    index, candidates = picker(heuristic=False)
    assert index._select_neighbours(candidates, 2) == [0, 1]           # both to the east


def test_heuristic_spreads_the_links():
    index, candidates = picker(heuristic=True)
    assert index._select_neighbours(candidates, 2) == [0, 3]           # one east, one west


def test_heuristic_can_keep_fewer_than_the_limit():
    index, candidates = picker(heuristic=True)
    assert index._select_neighbours(candidates[:3], 3) == [0]          # 1 and 2 are "behind" 0


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(13)
    xb = rng.standard_normal((1000, 16)).astype(np.float32)
    xq = rng.standard_normal((40, 16)).astype(np.float32)
    flat = FlatIndex("l2")
    flat.add(xb)
    _, gt = flat.search(xq, 10)
    return xb, xq, gt


def test_heuristic_graph_is_valid_and_smaller(data):
    xb, _, _ = data
    plain = HNSWIndex(M=8, ef_construction=64, heuristic=False)
    plain.add(xb)
    smart = HNSWIndex(M=8, ef_construction=64, heuristic=True)
    smart.add(xb)
    links = lambda ix: sum(len(v) for layer in ix.graph for v in layer.values())
    assert links(smart) < links(plain)
    for l, layer in enumerate(smart.graph):
        limit = smart.M0 if l == 0 else smart.M
        for node, nbrs in layer.items():
            assert len(nbrs) <= limit and node not in nbrs and all(n in layer for n in nbrs)


def test_heuristic_finds_more_with_few_links(data):
    xb, xq, gt = data
    recalls = {}
    for h in (False, True):
        index = HNSWIndex(M=4, ef_construction=64, ef_search=16, heuristic=h)
        index.add(xb)
        recalls[h] = recall_at_k(index.search(xq, 10)[1], gt, 10)
    assert recalls[True] > recalls[False], recalls


def test_the_setting_survives_save_and_load(data, tmp_path):
    xb, xq, _ = data
    for h in (False, True):
        index = HNSWIndex(M=6, ef_construction=40, heuristic=h)
        index.add(xb[:300])
        save(index, tmp_path / "h.npz")
        loaded = load(tmp_path / "h.npz")
        assert loaded.heuristic is h


@pytest.mark.skipif(not (DATA / "siftsmall_base.fvecs").exists(), reason="SIFT10K not downloaded")
def test_sift_m8_gap_with_faiss_closes():
    """At M = 8, plain selection reached 0.75 recall at ef_search = 10 and Faiss 0.88."""
    from vectorforge.io import load_sift
    xb, xq, _ = load_sift(DATA, "siftsmall")
    xb = xb[:3000]
    flat = FlatIndex("l2")
    flat.add(xb)
    _, gt = flat.search(xq, 10)
    recalls = {}
    for h in (False, True):
        index = HNSWIndex(M=8, ef_construction=100, ef_search=10, heuristic=h)
        index.add(xb)
        recalls[h] = recall_at_k(index.search(xq, 10)[1], gt, 10)
    assert recalls[True] >= recalls[False] + 0.05, recalls
