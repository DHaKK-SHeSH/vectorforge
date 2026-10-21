"""PQ: how much recall do you get for how many bytes per vector? (SIFT10K)

For m = 4, 8, 16, 32 pieces (always 256 entries per catalogue = 1 byte per piece)
it measures recall@10, queries per second (QPS), bytes per vector and training time for
  - your PQIndex (VectorForge)
  - Faiss's IndexPQ with the same m and nbits (if faiss is installed)
and the exact FlatIndex (512 bytes per vector) as the reference point.

Run from the project root:
    python scripts/sweep_pq.py

Training is the slow part: expect a few minutes in total on a laptop.

Outputs:
    docs/pq_sweep.csv   the numbers
    docs/pq_sweep.png   recall vs bytes per vector, and speed vs recall
"""

import csv
import time
from pathlib import Path

import numpy as np

from vectorforge.flat import FlatIndex
from vectorforge.io import load_sift, read_fvecs
from vectorforge.metrics import recall_at_k
from vectorforge.pq import PQIndex

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "siftsmall"
OUT = ROOT / "docs"
MS = [4, 8, 16, 32]
NBITS = 8
K = 10
REPEATS = 3


def timed_search(index, xq, k):
    """Best of REPEATS runs (less noise from whatever else the laptop is doing)."""
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
    print(f"base {xb.shape}  queries {xq.shape}  train {xt.shape}  nbits={NBITS}\n")
    rows = []   # (index, m, bytes_per_vector, recall, qps, train_s)

    # --- exact baseline: the full float32 vectors
    flat = FlatIndex("l2")
    flat.add(xb)
    ids, qps = timed_search(flat, xq, K)
    rows.append(("flat", "-", xb.nbytes / len(xb), recall_at_k(ids, gt, K), qps, 0.0))

    # --- your PQ
    for m in MS:
        pq = PQIndex(m=m, nbits=NBITS)
        t0 = time.perf_counter()
        pq.train(xt)
        train_s = time.perf_counter() - t0
        pq.add(xb)
        ids, qps = timed_search(pq, xq, K)
        bpv = pq.codes.nbytes / len(xb)
        rec = recall_at_k(ids, gt, K)
        rows.append(("vectorforge_pq", m, bpv, rec, qps, train_s))
        print(f"VectorForge PQ m={m:<3} trained in {train_s:5.1f}s  recall@10={rec:.3f}")

    # --- Faiss PQ, same settings (single thread, so the comparison is fair)
    try:
        import faiss
        faiss.omp_set_num_threads(1)
        for m in MS:
            fpq = faiss.IndexPQ(d, m, NBITS)
            t0 = time.perf_counter()
            fpq.train(xt)
            train_s = time.perf_counter() - t0
            fpq.add(xb)
            ids, qps = timed_search(fpq, xq, K)
            rec = recall_at_k(ids, gt, K)
            rows.append(("faiss_pq", m, fpq.pq.code_size, rec, qps, train_s))
            print(f"Faiss PQ       m={m:<3} trained in {train_s:5.1f}s  recall@10={rec:.3f}")
    except ImportError:
        print("faiss not installed: skipping the Faiss comparison (pip install faiss-cpu)")

    # --- print, save, plot
    print(f"\n{'index':<16}{'m':>4}{'bytes/vec':>11}{'recall@10':>11}{'QPS':>10}{'train s':>9}")
    for name, m, bpv, rec, qps, ts in rows:
        print(f"{name:<16}{str(m):>4}{bpv:>11.0f}{rec:>11.3f}{qps:>10.0f}{ts:>9.1f}")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "pq_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "m", "bytes_per_vector", "recall_at_10", "qps", "train_s"])
        for name, m, bpv, rec, qps, ts in rows:
            w.writerow([name, m, f"{bpv:.0f}", f"{rec:.4f}", f"{qps:.1f}", f"{ts:.2f}"])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed: skipping the plot")
        return

    styles = {"vectorforge_pq": ("VectorForge PQ", "o-"), "faiss_pq": ("Faiss IndexPQ", "s--")}
    flat_row = [r for r in rows if r[0] == "flat"][0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # left: what you pay in memory for each level of recall
    for key, (label, style) in styles.items():
        pts = [(bpv, rec, m) for n, m, bpv, rec, _, _ in rows if n == key]
        if not pts:
            continue
        bpv, rec, ms = zip(*pts)
        ax1.plot(bpv, rec, style, label=label)
        if key == "vectorforge_pq":
            for b, r, m in pts:
                ax1.annotate(f"m={m}", (b, r), textcoords="offset points", xytext=(5, -12), fontsize=7)
    ax1.plot(flat_row[2], flat_row[3], "k*", markersize=12, label="Flat (exact, float32)")
    ax1.set_xscale("log", base=2)
    ticks = [4, 8, 16, 32, 64, 128, 256, 512]
    ax1.set_xticks(ticks, [str(t) for t in ticks])
    ax1.set_xlabel("bytes per vector (log scale)")
    ax1.set_ylabel("recall@10")
    ax1.set_title("Memory vs recall")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    # right: speed vs recall
    for key, (label, style) in styles.items():
        pts = [(rec, qps, m) for n, m, _, rec, qps, _ in rows if n == key]
        if not pts:
            continue
        rec, qps, ms = zip(*pts)
        ax2.plot(rec, qps, style, label=label)
        if key == "vectorforge_pq":
            for r, q, m in pts:
                ax2.annotate(f"m={m}", (r, q), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax2.plot(flat_row[3], flat_row[4], "k*", markersize=12, label="Flat (exact)")
    ax2.set_yscale("log")
    ax2.set_xlabel("recall@10")
    ax2.set_ylabel("queries per second (log scale)")
    ax2.set_title("Speed vs recall")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    fig.suptitle(f"Product quantisation on SIFT10K (nbits={NBITS}, k={K})")
    fig.tight_layout()
    fig.savefig(OUT / "pq_sweep.png", dpi=150)
    print(f"\nsaved {OUT / 'pq_sweep.csv'} and {OUT / 'pq_sweep.png'}")


if __name__ == "__main__":
    main()
