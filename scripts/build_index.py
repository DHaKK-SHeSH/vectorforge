"""Build one VectorForge index on SIFT10K and save it to a file.

    python scripts/build_index.py --kind hnsw --out index.vf.npz
    python scripts/build_index.py --kind ivf --nlist 100 --nprobe 8 --out ivf.vf.npz

Then serve it:   VF_INDEX=index.vf.npz uvicorn vectorforge.service:app
(PowerShell:     $env:VF_INDEX = "index.vf.npz"; uvicorn vectorforge.service:app)
"""

import argparse
import time
from pathlib import Path

from vectorforge.flat import FlatIndex
from vectorforge.hnsw import HNSWIndex
from vectorforge.io import load_sift, read_fvecs
from vectorforge.ivf import IVFIndex
from vectorforge.ivfpq import IVFPQIndex
from vectorforge.metrics import recall_at_k
from vectorforge.persist import describe, load, save
from vectorforge.pq import PQIndex

DATA = Path(__file__).resolve().parents[1] / "data" / "siftsmall"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=["flat", "ivf", "pq", "ivfpq", "hnsw"], default="hnsw")
    p.add_argument("--out", default="index.vf.npz")
    p.add_argument("--nlist", type=int, default=100)
    p.add_argument("--nprobe", type=int, default=8)
    p.add_argument("--m", type=int, default=8, help="PQ pieces (bytes per vector)")
    p.add_argument("--M", type=int, default=16, help="HNSW links per node")
    p.add_argument("--ef-construction", type=int, default=100)
    p.add_argument("--ef-search", type=int, default=64)
    args = p.parse_args()

    xb, xq, gt = load_sift(DATA, "siftsmall")
    xt = read_fvecs(DATA / "siftsmall_learn.fvecs")

    index = {
        "flat": lambda: FlatIndex("l2"),
        "ivf": lambda: IVFIndex(nlist=args.nlist, nprobe=args.nprobe),
        "pq": lambda: PQIndex(m=args.m),
        "ivfpq": lambda: IVFPQIndex(nlist=args.nlist, nprobe=args.nprobe, m=args.m),
        "hnsw": lambda: HNSWIndex(M=args.M, ef_construction=args.ef_construction, ef_search=args.ef_search),
    }[args.kind]()

    t0 = time.perf_counter()
    if hasattr(index, "train"):
        index.train(xt)
    index.add(xb)
    print(f"built {type(index).__name__} in {time.perf_counter() - t0:.1f}s")

    save(index, args.out)
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"saved to {args.out} ({size_mb:.1f} MB)")

    # check the round trip: the loaded index must give the same answers
    t0 = time.perf_counter()
    loaded = load(args.out)
    load_s = time.perf_counter() - t0
    _, ids_before = index.search(xq, 10)
    _, ids_after = loaded.search(xq, 10)
    assert (ids_before == ids_after).all(), "loaded index gives different results"
    print(f"loaded back in {load_s:.2f}s, same results, recall@10 = {recall_at_k(ids_after, gt, 10):.3f}")
    print(describe(loaded))


if __name__ == "__main__":
    main()
