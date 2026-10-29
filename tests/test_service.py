import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from vectorforge import service  # noqa: E402
from vectorforge.hnsw import HNSWIndex  # noqa: E402
from vectorforge.ivf import IVFIndex  # noqa: E402
from vectorforge.persist import save  # noqa: E402


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(5)
    xb = rng.standard_normal((500, 8)).astype(np.float32)
    xq = rng.standard_normal((4, 8)).astype(np.float32)
    return xb, xq


@pytest.fixture
def client(data, tmp_path):
    xb, _ = data
    index = HNSWIndex(M=6, ef_construction=40, ef_search=20)
    index.add(xb)
    save(index, tmp_path / "h.npz")
    service.load_index(str(tmp_path / "h.npz"))
    yield TestClient(service.app), index
    service.STATE.update(index=None, info=None)


def test_health(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "HNSWIndex" and body["n_vectors"] == 500 and body["dim"] == 8


def test_search_matches_the_index(client, data):
    c, index = client
    _, xq = data
    r = c.post("/search", json={"vectors": xq.tolist(), "k": 5})
    assert r.status_code == 200
    body = r.json()
    _, ids = index.search(xq, 5)
    assert body["ids"] == ids.tolist()
    assert len(body["distances"]) == 4 and len(body["distances"][0]) == 5
    assert body["took_ms"] >= 0


def test_per_request_ef_search_does_not_change_the_default(client, data):
    c, index = client
    _, xq = data
    r = c.post("/search", json={"vectors": xq.tolist(), "k": 5, "ef_search": 200})
    assert r.status_code == 200
    assert index.ef_search == 20


def test_wrong_dimension_is_rejected(client):
    c, _ = client
    r = c.post("/search", json={"vectors": [[1.0, 2.0]], "k": 3})
    assert r.status_code == 422


def test_knob_for_another_index_type_is_rejected(client, data):
    c, _ = client
    _, xq = data
    r = c.post("/search", json={"vectors": xq.tolist(), "k": 3, "nprobe": 4})
    assert r.status_code == 422


def test_padding_becomes_minus_one(data, tmp_path):
    xb, xq = data
    index = IVFIndex(nlist=4, nprobe=1)
    index.train(xb)
    index.add(xb[:20])
    save(index, tmp_path / "ivf.npz")
    service.load_index(str(tmp_path / "ivf.npz"))
    try:
        r = TestClient(service.app).post("/search", json={"vectors": xq[:1].tolist(), "k": 50})
        assert r.status_code == 200
        ids, dists = r.json()["ids"][0], r.json()["distances"][0]
        assert -1 in ids and dists[ids.index(-1)] == -1.0
    finally:
        service.STATE.update(index=None, info=None)


def test_no_index_loaded_is_a_clear_error():
    service.STATE.update(index=None, info=None)
    r = TestClient(service.app).get("/health")
    assert r.status_code == 503
