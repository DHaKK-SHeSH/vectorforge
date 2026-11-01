"""v1.1 extra A: IVF-PQ + exact re-ranking on SIFT10K.

IVF-PQ (nlist = 100, m = 8 bytes per vector) gives a shortlist of k * k_factor
candidates; RefineIndex re-ranks them with exact distances to the original
vectors, which stay ON DISK (a memory-mapped .npy file): only the codes are in RAM.

For nprobe = 8 and 16 and k_factor = 1 ... 50 it measures recall@10 and QPS for
  - VectorForge RefineIndex(IVFPQIndex)
  - Faiss IndexRefineFlat(IndexIVFPQ), same settings (if faiss is installed)

Run from the project root:
    python scripts/sweep_refine.py

Outputs:
    docs/refine_sweep.csv   the numbers
    docs/refine_sweep.png   recall vs k_factor, and speed vs recall
"""

import csv
import tempfile
import time
from pathlib import Path

import numpy as np

from vectorforge.io import load_sift, read_fvecs
from vectorforge.ivfpq import IVFPQIndex
from vectorforge.metrics import recall_at_k
from vectorforge.refine import RefineIndex

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "siftsmall"
OUT = ROOT / "docs"
NLIST, M = 100, 8
NPROBES = [8, 16]
K_FACTORS = [1, 2, 5, 10, 20, 50]
K = 10
REPEATS = 3


def timed_search(index, xq, k):
    best, ids = float("inf"), None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        _, ids = index.search(xq, k)
        best = min(best, time.perf_counter() - t0)
    return ids, len(xq) / best


def main():
    xb, xq, gt = load_sift(DATA, "siftsmall")
    xt = read_fvecs(DATA / "siftsmall_learn.fvecs")
    d = xb.shape[1]
    rows = []   # (index, nprobe, k_factor, recall, qps)

    with tempfile.TemporaryDirectory() as tmp:
        np.save(Path(tmp) / "xb.npy", xb)
        xb_on_disk = np.load(Path(tmp) / "xb.npy", mmap_mode="r")      # full vectors stay on disk

        index = RefineIndex(IVFPQIndex(nlist=NLIST, m=M))
        index.train(xt)
        index.add(xb_on_disk)
        ram = sum(c.nbytes for c in index.base.list_codes) + 8 * len(xb) \
            + index.base.centroids.nbytes + index.base.pq.codebooks.nbytes
        print(f"in RAM: {ram / 1e6:.2f} MB (codes, ids, centroids, codebooks); "
              f"on disk: {xb.nbytes / 1e6:.2f} MB (full vectors)\n")
        for p in NPROBES:
            index.base.nprobe = p
            for kf in K_FACTORS:
                index.k_factor = kf
                ids, qps = timed_search(index, xq, K)
                rows.append(("vectorforge_refine", p, kf, recall_at_k(ids, gt, K), qps))
        del index, xb_on_disk

    try:
        import faiss
        faiss.omp_set_num_threads(1)
        base = faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, NLIST, M, 8)
        findex = faiss.IndexRefineFlat(base)
        findex.train(xt)
        findex.add(xb)
        for p in NPROBES:
            faiss.extract_index_ivf(findex).nprobe = p
            for kf in K_FACTORS:
                findex.k_factor = kf
                ids, qps = timed_search(findex, xq, K)
                rows.append(("faiss_refine", p, kf, recall_at_k(ids, gt, K), qps))
    except ImportError:
        print("faiss not installed: skipping the Faiss comparison (pip install faiss-cpu)")

    print(f"{'index':<20}{'nprobe':>7}{'k_factor':>9}{'recall@10':>11}{'QPS':>9}")
    for name, p, kf, rec, qps in rows:
        print(f"{name:<20}{p:>7}{kf:>9}{rec:>11.3f}{qps:>9.0f}")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "refine_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "nprobe", "k_factor", "recall_at_10", "qps"])
        for name, p, kf, rec, qps in rows:
            w.writerow([name, p, kf, f"{rec:.4f}", f"{qps:.1f}"])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed: skipping the plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    styles = {("vectorforge_refine", 8): ("VectorForge, nprobe=8", "o-"),
              ("vectorforge_refine", 16): ("VectorForge, nprobe=16", "^-"),
              ("faiss_refine", 8): ("Faiss, nprobe=8", "s--"),
              ("faiss_refine", 16): ("Faiss, nprobe=16", "v--")}
    for (key, p), (label, style) in styles.items():
        pts = [(kf, rec, qps) for n, pp, kf, rec, qps in rows if n == key and pp == p]
        if not pts:
            continue
        kfs, recs, qpss = zip(*pts)
        ax1.plot(kfs, recs, style, label=label)
        ax2.plot(recs, qpss, style, label=label)
    ax1.set_xscale("log")
    ax1.set_xticks(K_FACTORS, [str(k) for k in K_FACTORS])
    ax1.set_xlabel("k_factor (shortlist = 10 x k_factor)")
    ax1.set_ylabel("recall@10")
    ax1.set_title("Recall vs shortlist size (k_factor = 1: IVF-PQ alone)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)
    ax2.set_yscale("log")
    ax2.set_xlabel("recall@10")
    ax2.set_ylabel("queries per second (log scale)")
    ax2.set_title("Speed vs recall")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8)
    fig.suptitle(f"IVF-PQ + exact re-ranking on SIFT10K (nlist={NLIST}, {M} bytes per vector in RAM)")
    fig.tight_layout()
    fig.savefig(OUT / "refine_sweep.png", dpi=150)
    print(f"\nsaved {OUT / 'refine_sweep.csv'} and {OUT / 'refine_sweep.png'}")


if __name__ == "__main__":
    main()
