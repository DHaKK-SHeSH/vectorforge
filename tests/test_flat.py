"""Your FlatIndex is checked against a deliberately slow, obviously-correct loop."""

import numpy as np
import pytest

from vectorforge.flat import FlatIndex


def naive_knn(xb, xq, k, metric):
    dists = np.empty((len(xq), len(xb)), dtype=np.float64)
    for i, q in enumerate(xq.astype(np.float64)):
        for j, x in enumerate(xb.astype(np.float64)):
            dists[i, j] = np.sum((q - x) ** 2) if metric == "l2" else np.dot(q, x)
    order = np.argsort(dists if metric == "l2" else -dists, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(dists, order, axis=1), order


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    xb = rng.standard_normal((300, 32)).astype(np.float32)
    xq = rng.standard_normal((20, 32)).astype(np.float32)
    return xb, xq


@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_matches_naive(data, metric):
    xb, xq = data
    index = FlatIndex(metric)
    index.add(xb)
    d, ids = index.search(xq, k=10)
    d_ref, ids_ref = naive_knn(xb, xq, 10, metric)
    assert ids.shape == (20, 10) and d.shape == (20, 10)
    np.testing.assert_array_equal(ids, ids_ref)
    np.testing.assert_allclose(d, d_ref, rtol=1e-4, atol=1e-3)


def test_l2_sorted_ascending(data):
    xb, xq = data
    index = FlatIndex("l2")
    index.add(xb)
    d, _ = index.search(xq, k=10)
    assert np.all(np.diff(d, axis=1) >= 0)


def test_ip_sorted_descending(data):
    xb, xq = data
    index = FlatIndex("ip")
    index.add(xb)
    d, _ = index.search(xq, k=10)
    assert np.all(np.diff(d, axis=1) <= 0)


def test_query_equal_to_base_vector_is_its_own_nearest(data):
    xb, _ = data
    index = FlatIndex("l2")
    index.add(xb)
    d, ids = index.search(xb[:5], k=1)
    np.testing.assert_array_equal(ids[:, 0], np.arange(5))
    assert np.all(d[:, 0] >= 0), "squared distances must never be negative"
    np.testing.assert_allclose(d[:, 0], 0.0, atol=1e-3)
