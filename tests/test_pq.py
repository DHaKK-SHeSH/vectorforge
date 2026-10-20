from pathlib import Path

import numpy as np
import pytest

from vectorforge.pq import PQIndex

DATA = Path(__file__).resolve().parents[1] / "data" / "siftsmall"


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(5)
    xt = rng.standard_normal((1500, 16)).astype(np.float32)
    xb = rng.standard_normal((800, 16)).astype(np.float32)
    xq = rng.standard_normal((20, 16)).astype(np.float32)
    return xt, xb, xq


@pytest.fixture(scope="module")
def trained(data):
    xt, xb, _ = data
    pq = PQIndex(m=4, nbits=4)          # 4 pieces of 4 numbers, 16 centroids each: fast to train
    pq.train(xt)
    pq.add(xb)
    return pq


# ---------------------------------------------------------------- train
def test_codebook_shape(data):
    xt, *_ = data
    pq = PQIndex(m=4, nbits=4)          # train only: works before encode/add are written
    pq.train(xt)
    assert pq.dsub == 4
    assert pq.codebooks.shape == (4, 16, 4)
    assert pq.codebooks.dtype == np.float32


def test_dimension_must_divide_into_pieces(data):
    xt, *_ = data
    with pytest.raises(ValueError):
        PQIndex(m=5, nbits=4).train(xt)    # 16 numbers can't be cut into 5 equal pieces


# ---------------------------------------------------------------- encode / decode
def test_codes_are_small_bytes(trained, data):
    _, xb, _ = data
    assert trained.codes.shape == (800, 4)
    assert trained.codes.dtype == np.uint8
    assert trained.codes.max() < 16
    assert trained.codes.nbytes == 800 * 4      # 4 bytes per vector instead of 64


def test_encode_picks_nearest_centroid_per_piece(trained, data):
    _, xb, _ = data
    codes = trained.encode(xb[:30])
    for i in range(30):
        for j in range(4):
            piece = xb[i, j * 4:(j + 1) * 4]
            d = ((trained.codebooks[j] - piece) ** 2).sum(axis=1)
            assert codes[i, j] == d.argmin()


def test_decode_glues_the_right_centroids(trained):
    codes = np.array([[0, 1, 2, 3], [15, 14, 13, 12]], dtype=np.uint8)
    x = trained.decode(codes)
    assert x.shape == (2, 16)
    for i in range(2):
        for j in range(4):
            np.testing.assert_allclose(x[i, j * 4:(j + 1) * 4], trained.codebooks[j][codes[i, j]])


def test_reconstruction_beats_predicting_the_mean(trained, data):
    _, xb, _ = data
    approx = trained.decode(trained.encode(xb))
    err_pq = ((xb - approx) ** 2).sum(axis=1).mean()
    err_mean = ((xb - xb.mean(axis=0)) ** 2).sum(axis=1).mean()
    assert err_pq < 0.7 * err_mean


# ---------------------------------------------------------------- search
def test_search_distances_equal_distance_to_decoded_vectors(trained, data):
    """The lookup-table trick must give EXACTLY ||q - decode(code)||^2."""
    _, xb, xq = data
    d, ids = trained.search(xq, 10)
    approx = trained.decode(trained.codes)
    for q in range(len(xq)):
        true = ((approx[ids[q]] - xq[q]) ** 2).sum(axis=1)
        np.testing.assert_allclose(d[q], true, rtol=1e-4, atol=1e-3)


def test_search_returns_the_nearest_decoded_vectors(trained, data):
    _, _, xq = data
    approx = trained.decode(trained.codes)
    _, ids = trained.search(xq, 5)
    for q in range(len(xq)):
        all_d = ((approx - xq[q]) ** 2).sum(axis=1)
        assert all_d[ids[q]].max() <= np.sort(all_d)[4] + 1e-3


def test_search_sorted_and_shaped(trained, data):
    _, _, xq = data
    d, ids = trained.search(xq, 10)
    assert d.shape == (20, 10) and ids.shape == (20, 10)
    assert np.all(np.diff(d, axis=1) >= -1e-5)


# ---------------------------------------------------------------- real data
@pytest.mark.skipif(not (DATA / "siftsmall_base.fvecs").exists(), reason="SIFT10K not downloaded")
def test_sift_recall_and_compression():
    from vectorforge.io import load_sift, read_fvecs
    from vectorforge.metrics import recall_at_k
    xb, xq, gt = load_sift(DATA, "siftsmall")
    xt = read_fvecs(DATA / "siftsmall_learn.fvecs")
    pq = PQIndex(m=8, nbits=8)
    pq.train(xt)
    pq.add(xb)
    _, ids = pq.search(xq, 10)
    assert pq.codes.nbytes * 64 == xb.nbytes         # 64x smaller than the float32 vectors
    assert recall_at_k(ids, gt, 10) >= 0.35            # approximate, but far better than chance
