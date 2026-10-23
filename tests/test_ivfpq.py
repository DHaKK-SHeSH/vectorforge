from pathlib import Path

import numpy as np
import pytest

from vectorforge.ivfpq import IVFPQIndex
from vectorforge.kmeans import assign

DATA = Path(__file__).resolve().parents[1] / "data" / "siftsmall"


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    xt = rng.standard_normal((3000, 16)).astype(np.float32)
    xb = rng.standard_normal((1500, 16)).astype(np.float32)
    xq = rng.standard_normal((25, 16)).astype(np.float32)
    return xt, xb, xq


def make(xt, xb, by_residual=True):
    # 10 cells; PQ with 4 pieces of 4 numbers and 16 entries per catalogue: fast to train
    index = IVFPQIndex(nlist=10, nprobe=1, m=4, nbits=4, by_residual=by_residual)
    index.train(xt)
    index.add(xb)
    return index


@pytest.fixture(scope="module")
def built(data):
    xt, xb, _ = data
    return make(xt, xb)


def reconstruct(index, xb):
    """What the index 'thinks' each base vector is: cell centre + decoded offset."""
    approx = np.zeros_like(xb)
    for c, ids in enumerate(index.lists):
        if len(ids):
            approx[ids] = index.pq.decode(index.list_codes[c])
            if index.by_residual:
                approx[ids] += index.centroids[c]
    return approx


# ---------------------------------------------------------------- train
def test_train_learns_cells_and_codebooks(data):
    xt, *_ = data
    index = IVFPQIndex(nlist=10, m=4, nbits=4)
    index.train(xt)
    assert index.centroids.shape == (10, 16)
    assert index.pq.codebooks.shape == (4, 16, 4)
    assert index.is_trained


def test_add_before_train_raises(data):
    _, xb, _ = data
    with pytest.raises(Exception):
        IVFPQIndex(nlist=10, m=4, nbits=4).add(xb)


# ---------------------------------------------------------------- add
def test_every_vector_in_exactly_one_list(built, data):
    _, xb, _ = data
    assert len(built.lists) == 10 and len(built.list_codes) == 10
    all_ids = np.sort(np.concatenate(built.lists))
    np.testing.assert_array_equal(all_ids, np.arange(len(xb)))
    for ids, codes in zip(built.lists, built.list_codes):
        assert codes.shape == (len(ids), 4)
        assert codes.dtype == np.uint8


def test_vectors_sit_in_their_nearest_cell(built, data):
    _, xb, _ = data
    labels, _ = assign(xb, built.centroids)
    for c, ids in enumerate(built.lists):
        assert np.all(labels[ids] == c)


def test_codes_store_the_offset_from_the_cell_centre(built, data):
    _, xb, _ = data
    for c, ids in enumerate(built.lists):
        if len(ids):
            expected = built.pq.encode(xb[ids] - built.centroids[c])
            np.testing.assert_array_equal(built.list_codes[c], expected)


def test_without_residuals_codes_store_the_vector_itself(data):
    xt, xb, _ = data
    index = make(xt, xb, by_residual=False)
    for c, ids in enumerate(index.lists):
        if len(ids):
            np.testing.assert_array_equal(index.list_codes[c], index.pq.encode(xb[ids]))


# ---------------------------------------------------------------- search
def test_search_distances_equal_distance_to_reconstructions(built, data):
    """Table lookups must give EXACTLY ||q - (cell centre + decoded offset)||^2."""
    _, xb, xq = data
    built.nprobe = 10
    d, ids = built.search(xq, 10)
    approx = reconstruct(built, xb)
    for q in range(len(xq)):
        true = ((approx[ids[q]] - xq[q]) ** 2).sum(axis=1)
        np.testing.assert_allclose(d[q], true, rtol=1e-4, atol=1e-3)


def test_full_probe_returns_the_nearest_reconstructions(built, data):
    _, xb, xq = data
    built.nprobe = 10                       # every cell: nothing is skipped
    _, ids = built.search(xq, 5)
    approx = reconstruct(built, xb)
    for q in range(len(xq)):
        all_d = ((approx - xq[q]) ** 2).sum(axis=1)
        assert all_d[ids[q]].max() <= np.sort(all_d)[4] + 1e-3


def test_probing_more_cells_never_makes_the_kth_result_worse(built, data):
    """More cells = more candidates, so the 10th-best distance can only shrink."""
    _, _, xq = data
    prev = None
    for nprobe in (1, 2, 4, 10):
        built.nprobe = nprobe
        d, _ = built.search(xq, 10)
        kth = d[:, -1]
        if prev is not None:
            assert np.all(kth <= prev + 1e-4)
        prev = kth


def test_results_sorted_and_padded(built, data):
    _, _, xq = data
    built.nprobe = 1
    d, i = built.search(xq, 500)            # more than one cell holds: forces padding
    valid = np.isfinite(d)
    assert (~valid).any()
    assert np.all(i[~valid] == -1)
    for row in d:
        r = row[np.isfinite(row)]
        assert np.all(np.diff(r) >= -1e-5)


# ---------------------------------------------------------------- real data
@pytest.mark.skipif(not (DATA / "siftsmall_base.fvecs").exists(), reason="SIFT10K not downloaded")
def test_sift_residuals_beat_raw_vectors():
    """Slow (trains two indexes on SIFT): about a minute on a laptop."""
    from vectorforge.io import load_sift, read_fvecs
    from vectorforge.metrics import recall_at_k
    xb, xq, gt = load_sift(DATA, "siftsmall")
    xt = read_fvecs(DATA / "siftsmall_learn.fvecs")
    recalls = {}
    for by_residual in (True, False):
        index = IVFPQIndex(nlist=100, nprobe=8, m=8, nbits=8, by_residual=by_residual)
        index.train(xt)
        index.add(xb)
        assert sum(c.nbytes for c in index.list_codes) == len(xb) * 8   # 8 bytes per vector
        _, ids = index.search(xq, 10)
        recalls[by_residual] = recall_at_k(ids, gt, 10)
    assert recalls[True] >= 0.5, recalls
    assert recalls[True] > recalls[False], recalls
