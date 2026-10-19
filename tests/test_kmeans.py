import numpy as np
import pytest

from vectorforge.kmeans import assign, init_kmeanspp, kmeans, update_centroids


@pytest.fixture
def blobs():
    """Three tight, far-apart groups of 100 points each in 8 dimensions."""
    rng = np.random.default_rng(0)
    centers = np.array([[0.0] * 8, [50.0] * 8, [-50.0] + [50.0] * 7], dtype=np.float32)
    x = np.concatenate([c + rng.standard_normal((100, 8)).astype(np.float32) for c in centers])
    truth = np.repeat(np.arange(3), 100)
    return x, truth


# ---------------------------------------------------------------- assign
def test_assign_matches_slow_loop():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((200, 16)).astype(np.float32)
    c = rng.standard_normal((7, 16)).astype(np.float32)
    labels, d = assign(x, c)
    slow = np.array([[np.sum((p - q) ** 2) for q in c] for p in x])
    np.testing.assert_array_equal(labels, slow.argmin(axis=1))
    np.testing.assert_allclose(d, slow.min(axis=1), rtol=1e-4, atol=1e-3)
    assert labels.shape == (200,) and d.shape == (200,)


def test_assign_distances_never_negative():
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    _, d = assign(x, x.copy())
    assert np.all(d >= 0)


# ---------------------------------------------------------------- update
def test_update_is_mean_of_members():
    x = np.array([[0, 0], [2, 0], [10, 10], [12, 10]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1])
    old = np.zeros((2, 2), dtype=np.float32)
    new = update_centroids(x, labels, old, np.random.default_rng(0))
    np.testing.assert_allclose(new, [[1, 0], [11, 10]])


def test_empty_cluster_is_reseeded_not_nan():
    x = np.array([[0, 0], [2, 0], [10, 10]], dtype=np.float32)
    labels = np.array([0, 0, 0])  # clusters 1 and 2 are empty
    old = np.zeros((3, 2), dtype=np.float32)
    new = update_centroids(x, labels, old, np.random.default_rng(0))
    assert np.all(np.isfinite(new))
    np.testing.assert_allclose(new[0], [4, 10 / 3], rtol=1e-5)
    for j in (1, 2):  # re-seeded to an actual data point
        assert any(np.allclose(new[j], p) for p in x)


# ---------------------------------------------------------------- k-means++
def test_kmeanspp_returns_k_distinct_data_points(blobs):
    x, _ = blobs
    c = init_kmeanspp(x, 3, np.random.default_rng(0))
    assert c.shape == (3, 8)
    for row in c:
        assert any(np.array_equal(row, p) for p in x)
    assert len({tuple(row) for row in c}) == 3


def test_kmeanspp_spreads_over_separated_blobs(blobs):
    # with blobs this far apart, k-means++ should almost always pick one seed per blob
    x, truth = blobs
    hits = 0
    for seed in range(20):
        c = init_kmeanspp(x, 3, np.random.default_rng(seed))
        labels, _ = assign(c, np.array([x[truth == g].mean(axis=0) for g in range(3)]))
        hits += len(set(labels.tolist())) == 3
    assert hits >= 19


# ---------------------------------------------------------------- the full loop
@pytest.mark.parametrize("init", ["random", "kmeans++"])
def test_inertia_never_increases(init):
    rng = np.random.default_rng(2)
    x = rng.standard_normal((500, 10)).astype(np.float32)
    _, _, history = kmeans(x, k=16, n_iter=30, init=init, seed=0)
    assert len(history) >= 2
    diffs = np.diff(history)
    assert np.all(diffs <= 1e-3 * history[0]), "inertia went up — check assign/update"


def test_recovers_separated_blobs(blobs):
    x, truth = blobs
    centroids, labels, _ = kmeans(x, k=3, init="kmeans++", seed=0)
    assert centroids.shape == (3, 8)
    # every true group should map to exactly one cluster
    for g in range(3):
        assert len(np.unique(labels[truth == g])) == 1
    assert len(np.unique(labels)) == 3


def test_same_seed_same_result(blobs):
    x, _ = blobs
    c1, _, _ = kmeans(x, k=5, seed=7)
    c2, _, _ = kmeans(x, k=5, seed=7)
    np.testing.assert_array_equal(c1, c2)
