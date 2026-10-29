"""Save a built index to disk and load it back, without rebuilding it.

    from vectorforge.persist import save, load
    save(index, "index.vf.npz")
    index = load("index.vf.npz")      # same results as before, instantly

Works for FlatIndex, IVFIndex, PQIndex, IVFPQIndex and HNSWIndex.

Format: ONE .npz file (NumPy's zip of arrays). The settings go in a small JSON
string stored inside it; everything else is plain arrays. Lists of arrays with
different lengths (IVF's inverted lists, HNSW's neighbour lists) are stored the
usual way: all values glued into one flat array + an "offsets" array saying
where each piece starts and ends.

Loading never uses pickle (allow_pickle=False), so opening a file from someone
else can't run code on your machine.
"""

import json
from pathlib import Path

import numpy as np

from .flat import FlatIndex
from .hnsw import HNSWIndex
from .ivf import IVFIndex
from .ivfpq import IVFPQIndex
from .pq import PQIndex

FORMAT_VERSION = 1


# ---------------------------------------------------------------- ragged lists
def _pack(arrays, dtype):
    """[array, array, ...] of different lengths -> (flat values, offsets).
    Piece i is values[offsets[i]:offsets[i + 1]]."""
    lengths = [len(a) for a in arrays]
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    if arrays and sum(lengths):
        values = np.concatenate([np.asarray(a, dtype=dtype) for a in arrays])
    else:
        values = np.zeros((0,) + (np.asarray(arrays[0]).shape[1:] if arrays else ()), dtype=dtype)
    return values, offsets


def _unpack(values, offsets):
    return [values[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


# ---------------------------------------------------------------- save
def save(index, path) -> None:
    """Write a built index to `path` (a .npz file)."""
    kind = type(index).__name__
    meta = {"format": FORMAT_VERSION, "kind": kind}
    arrays = {}

    if kind == "FlatIndex":
        meta["metric"] = index.metric
        arrays["xb"] = index.xb

    elif kind == "IVFIndex":
        meta.update(nlist=index.nlist, nprobe=index.nprobe, n_iter=index.n_iter, seed=index.seed)
        arrays["centroids"] = index.centroids
        arrays["xb"] = index.xb
        arrays["list_ids"], arrays["list_offsets"] = _pack(index.lists, np.int64)

    elif kind == "PQIndex":
        meta.update(m=index.m, nbits=index.nbits, n_iter=index.n_iter, seed=index.seed, dsub=index.dsub)
        arrays["codebooks"] = index.codebooks
        arrays["codes"] = index.codes

    elif kind == "IVFPQIndex":
        pq = index.pq
        meta.update(nlist=index.nlist, nprobe=index.nprobe, by_residual=index.by_residual,
                    n_iter=index.n_iter, seed=index.seed, m=pq.m, nbits=pq.nbits, dsub=pq.dsub)
        arrays["centroids"] = index.centroids
        arrays["codebooks"] = pq.codebooks
        arrays["list_ids"], arrays["list_offsets"] = _pack(index.lists, np.int64)
        arrays["list_codes"], _ = _pack(index.list_codes, np.uint8)

    elif kind == "HNSWIndex":
        meta.update(M=index.M, ef_construction=index.ef_construction, ef_search=index.ef_search,
                    entry_point=index.entry_point, max_level=index.max_level,
                    rng_state=index.rng.bit_generator.state)       # so later inserts continue identically
        arrays["xb"] = index.xb
        arrays["levels"] = np.asarray(index.levels, dtype=np.int64)
        for l, layer in enumerate(index.graph):
            nodes = list(layer)                                     # keep the dict's order
            arrays[f"layer{l}_nodes"] = np.asarray(nodes, dtype=np.int64)
            arrays[f"layer{l}_links"], arrays[f"layer{l}_offsets"] = _pack([layer[i] for i in nodes], np.int64)
        meta["n_layers"] = len(index.graph)

    else:
        raise TypeError(f"don't know how to save a {kind}")

    for name, value in arrays.items():
        if value is None:
            raise ValueError(f"{kind}.{name} is empty: train and add before saving")
    np.savez(Path(path), meta=np.array(json.dumps(meta)), **arrays)


# ---------------------------------------------------------------- load
def load(path):
    """Read an index written by save(). Returns a ready-to-search index."""
    with np.load(Path(path), allow_pickle=False) as f:
        meta = json.loads(str(f["meta"]))
        a = {k: f[k] for k in f.files if k != "meta"}
    if meta.get("format") != FORMAT_VERSION:
        raise ValueError(f"unsupported file format {meta.get('format')}")
    kind = meta["kind"]

    if kind == "FlatIndex":
        index = FlatIndex(meta["metric"])
        index.add(a["xb"])                                          # recomputes the norms

    elif kind == "IVFIndex":
        index = IVFIndex(nlist=meta["nlist"], nprobe=meta["nprobe"], n_iter=meta["n_iter"], seed=meta["seed"])
        index.centroids = a["centroids"]
        index.xb = a["xb"]
        index.lists = _unpack(a["list_ids"], a["list_offsets"])

    elif kind == "PQIndex":
        index = PQIndex(m=meta["m"], nbits=meta["nbits"], n_iter=meta["n_iter"], seed=meta["seed"])
        index.dsub = meta["dsub"]
        index.codebooks = a["codebooks"]
        index.codes = a["codes"]

    elif kind == "IVFPQIndex":
        index = IVFPQIndex(nlist=meta["nlist"], nprobe=meta["nprobe"], m=meta["m"], nbits=meta["nbits"],
                           by_residual=meta["by_residual"], n_iter=meta["n_iter"], seed=meta["seed"])
        index.centroids = a["centroids"]
        index.pq.dsub = meta["dsub"]
        index.pq.codebooks = a["codebooks"]
        index.lists = _unpack(a["list_ids"], a["list_offsets"])
        index.list_codes = _unpack(a["list_codes"], a["list_offsets"])

    elif kind == "HNSWIndex":
        index = HNSWIndex(M=meta["M"], ef_construction=meta["ef_construction"], ef_search=meta["ef_search"])
        index.rng.bit_generator.state = meta["rng_state"]
        index.xb = a["xb"]
        index.levels = [int(v) for v in a["levels"]]
        index.graph = []
        for l in range(meta["n_layers"]):
            nodes = a[f"layer{l}_nodes"].tolist()
            links = _unpack(a[f"layer{l}_links"], a[f"layer{l}_offsets"])
            index.graph.append({node: nbrs.tolist() for node, nbrs in zip(nodes, links)})
        index.entry_point = meta["entry_point"]
        index.max_level = meta["max_level"]

    else:
        raise ValueError(f"unknown index kind {kind!r}")
    return index


def describe(index) -> dict:
    """A small summary for logs and the /health endpoint."""
    kind = type(index).__name__
    if kind == "PQIndex":
        n, d = len(index.codes), index.m * index.dsub
    elif kind == "IVFPQIndex":
        n, d = sum(len(ids) for ids in index.lists), index.centroids.shape[1]
    else:
        n, d = index.xb.shape
    info = {"kind": kind, "n_vectors": int(n), "dim": int(d)}
    for knob in ("nprobe", "ef_search"):
        if hasattr(index, knob):
            info[knob] = getattr(index, knob)
    return info
