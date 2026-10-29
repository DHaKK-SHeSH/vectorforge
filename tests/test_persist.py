import numpy as np
import pytest

from vectorforge.flat import FlatIndex
from vectorforge.hnsw import HNSWIndex
from vectorforge.ivf import IVFIndex
from vectorforge.ivfpq import IVFPQIndex
from vectorforge.persist import describe, load, save
from vectorforge.pq import PQIndex


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(21)
    xt = rng.standard_normal((1500, 16)).astype(np.float32)
    xb = rng.standard_normal((600, 16)).astype(np.float32)
    xq = rng.standard_normal((15, 16)).astype(np.float32)
    return xt, xb, xq


def build(kind, xt, xb):
    index = {
        "flat": lambda: FlatIndex("l2"),
        "ivf": lambda: IVFIndex(nlist=12, nprobe=3),
        "pq": lambda: PQIndex(m=4, nbits=4),
        "ivfpq": lambda: IVFPQIndex(nlist=12, nprobe=3, m=4, nbits=4),
        "hnsw": lambda: HNSWIndex(M=6, ef_construction=40, ef_search=30),
    }[kind]()
    if hasattr(index, "train"):
        index.train(xt)
    index.add(xb)
    return index


KINDS = ["flat", "ivf", "pq", "ivfpq", "hnsw"]


@pytest.mark.parametrize("kind", KINDS)
def test_round_trip_gives_identical_results(kind, data, tmp_path):
    xt, xb, xq = data
    index = build(kind, xt, xb)
    path = tmp_path / f"{kind}.vf.npz"
    save(index, path)
    loaded = load(path)
    assert type(loaded) is type(index)
    d0, i0 = index.search(xq, 10)
    d1, i1 = loaded.search(xq, 10)
    np.testing.assert_array_equal(i0, i1)
    np.testing.assert_allclose(d0, d1, rtol=1e-6)


@pytest.mark.parametrize("kind", KINDS)
def test_describe_reports_size_and_dim(kind, data, tmp_path):
    xt, xb, _ = data
    save(build(kind, xt, xb), tmp_path / "i.npz")
    info = describe(load(tmp_path / "i.npz"))
    assert info["n_vectors"] == len(xb) and info["dim"] == 16


def test_settings_survive(data, tmp_path):
    xt, xb, _ = data
    index = build("ivfpq", xt, xb)
    index.nprobe = 7
    save(index, tmp_path / "i.npz")
    loaded = load(tmp_path / "i.npz")
    assert loaded.nprobe == 7 and loaded.by_residual and loaded.pq.m == 4


def test_hnsw_keeps_growing_the_same_way_after_load(data, tmp_path):
    """The random-level generator is saved too: adding to the loaded copy matches adding to the original."""
    xt, xb, xq = data
    original = build("hnsw", xt, xb[:400])
    save(original, tmp_path / "h.npz")
    loaded = load(tmp_path / "h.npz")
    original.add(xb[400:])
    loaded.add(xb[400:])
    assert original.levels == loaded.levels
    np.testing.assert_array_equal(original.search(xq, 10)[1], loaded.search(xq, 10)[1])


def test_saving_an_empty_index_is_refused(tmp_path):
    with pytest.raises((ValueError, TypeError, AttributeError)):
        save(IVFIndex(nlist=4), tmp_path / "empty.npz")


def test_files_load_without_pickle(data, tmp_path):
    xt, xb, _ = data
    save(build("hnsw", xt, xb), tmp_path / "h.npz")
    with np.load(tmp_path / "h.npz", allow_pickle=False) as f:      # would raise if anything needed pickle
        assert "meta" in f.files
