"""A small HTTP search service around a saved VectorForge index.

    pip install -e ".[service]"
    python scripts/build_index.py --kind hnsw --out index.vf.npz     # build + save once
    VF_INDEX=index.vf.npz uvicorn vectorforge.service:app --port 8000

Then:
    GET  /health   -> what is loaded: {"kind": "HNSWIndex", "n_vectors": 10000, "dim": 128, ...}
    POST /search   -> {"vectors": [[...128 numbers...], ...], "k": 10, "ef_search": 64}
                      returns {"ids": [[...]], "distances": [[...]], "took_ms": 1.9}
    GET  /docs     -> interactive API page (FastAPI makes it automatically)

The index is loaded once at start-up and is read-only: the service only searches.
(PowerShell: set the variable with  $env:VF_INDEX = "index.vf.npz"  before uvicorn.)
"""

import os
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .persist import describe, load

STATE = {"index": None, "info": None}
LOCK = threading.Lock()          # one search at a time: nprobe / ef_search are set on the shared index


class SearchRequest(BaseModel):
    vectors: List[List[float]] = Field(..., min_length=1, max_length=1000,
                                       description="query vectors, one list of numbers per query")
    k: int = Field(10, ge=1, le=1000, description="how many neighbours per query")
    nprobe: Optional[int] = Field(None, ge=1, description="IVF / IVF-PQ only: cells to visit")
    ef_search: Optional[int] = Field(None, ge=1, description="HNSW only: shortlist size")


class SearchResponse(BaseModel):
    ids: List[List[int]]
    distances: List[List[float]]
    took_ms: float


def load_index(path: str) -> None:
    STATE["index"] = load(path)
    STATE["info"] = describe(STATE["index"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.environ.get("VF_INDEX")
    if path:
        if not os.path.exists(path):
            raise RuntimeError(f"VF_INDEX points to {path!r}, which doesn't exist. Build it first: "
                               f"python scripts/build_index.py --out {path}")
        load_index(path)
    yield


app = FastAPI(title="VectorForge", description="Nearest-neighbour search from scratch.", lifespan=lifespan)


@app.get("/health")
def health():
    if STATE["index"] is None:
        raise HTTPException(503, "no index loaded: set VF_INDEX to a file written by vectorforge.persist.save")
    return {"status": "ok", **STATE["info"]}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    index, info = STATE["index"], STATE["info"]
    if index is None:
        raise HTTPException(503, "no index loaded")

    xq = np.asarray(req.vectors, dtype=np.float32)
    if xq.ndim != 2 or xq.shape[1] != info["dim"]:
        raise HTTPException(422, f"every query needs exactly {info['dim']} numbers")
    for knob in ("nprobe", "ef_search"):
        if getattr(req, knob) is not None and not hasattr(index, knob):
            raise HTTPException(422, f"{knob} does not apply to a {info['kind']}")

    with LOCK:
        old = {knob: getattr(index, knob) for knob in ("nprobe", "ef_search") if hasattr(index, knob)}
        try:
            for knob in old:
                if getattr(req, knob) is not None:
                    setattr(index, knob, getattr(req, knob))
            t0 = time.perf_counter()
            D, I = index.search(xq, req.k)
            took = (time.perf_counter() - t0) * 1000
        finally:
            for knob, value in old.items():               # requests never change the defaults
                setattr(index, knob, value)

    D = np.where(np.isfinite(D), D, -1.0)                 # JSON has no infinity: padded slots get -1
    return SearchResponse(ids=I.tolist(), distances=D.tolist(), took_ms=round(took, 3))
