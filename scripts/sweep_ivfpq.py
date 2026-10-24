"""IVF-PQ: speed vs recall over nprobe, with and without the residual trick (SIFT10K).

For nprobe = 1 ... 100 (nlist = 100, m = 8, 8 bits = 8 bytes per vector) it measures
recall@10 and queries per second (QPS) for
  - your IVFPQIndex with residuals      (by_residual=True, what Faiss does)
  - your IVFPQIndex without residuals   (by_residual=False: PQ on the raw vectors)
  - Faiss's IndexIVFPQ, same settings   (if faiss is installed)
and, as reference points, exact Flat search and your plain PQ with m = 8.

Run from the project root:
    python scripts/sweep_ivfpq.py

Training is the slow part (two IVF-PQ indexes + one PQ): expect a few minutes.

Outputs:
    docs/ivfpq_sweep.csv   the numbers
    docs/ivfpq_sweep.png   recall vs QPS
"""

import csv
import time
from pathlib import Path

from vectorforge.flat import FlatIndex
from vectorforge.io import load_sift, read_fvecs
from vectorforge.ivfpq import IVFPQIndex
from vectorforge.metrics import recall_at_k
from vectorforge.pq import PQIndex

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "siftsmall"
OUT = ROOT / "docs"
NLIST = 100
M = 8
NBITS = 8
NPROBES = [1, 2, 4, 8, 16, 32, 100]
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
    print(f"base {xb.shape}  queries {xq.shape}  train {xt.shape}  nlist={NLIST}  m={M}\n")
    rows = []   # (index, nprobe, recall, qps, train_s)

    # --- reference points
    flat = FlatIndex("l2")
    flat.add(xb)
    ids, qps = timed_search(flat, xq, K)
    rows.append(("flat", "-", recall_at_k(ids, gt, K), qps, 0.0))

    pq = PQIndex(m=M, nbits=NBITS)
    t0 = time.perf_counter()
    pq.train(xt)
    train_s = time.perf_counter() - t0
    pq.add(xb)
    ids, qps = timed_search(pq, xq, K)
    rows.append(("vectorforge_pq", "-", recall_at_k(ids, gt, K), qps, train_s))
    print(f"VectorForge PQ (m={M}) trained in {train_s:.1f}s")

    # --- your IVF-PQ, with and without residuals
    for by_residual, name in ((True, "vectorforge_ivfpq"), (False, "vectorforge_ivfpq_raw")):
        index = IVFPQIndex(nlist=NLIST, m=M, nbits=NBITS, by_residual=by_residual)
        t0 = time.perf_counter()
        index.train(xt)
        train_s = time.perf_counter() - t0
        index.add(xb)
        code_bytes = sum(c.nbytes for c in index.list_codes)
        print(f"{name}: trained in {train_s:.1f}s, codes = {code_bytes / len(xb):.0f} bytes/vector")
        for nprobe in NPROBES:
            index.nprobe = nprobe
            ids, qps = timed_search(index, xq, K)
            rows.append((name, nprobe, recall_at_k(ids, gt, K), qps, train_s))

    # --- Faiss IVF-PQ, same settings (single thread, so the comparison is fair)
    try:
        import faiss
        faiss.omp_set_num_threads(1)
        quantizer = faiss.IndexFlatL2(d)
        fivfpq = faiss.IndexIVFPQ(quantizer, d, NLIST, M, NBITS)
        t0 = time.perf_counter()
        fivfpq.train(xt)
        train_s = time.perf_counter() - t0
        fivfpq.add(xb)
        for nprobe in NPROBES:
            fivfpq.nprobe = nprobe
            ids, qps = timed_search(fivfpq, xq, K)
            rows.append(("faiss_ivfpq", nprobe, recall_at_k(ids, gt, K), qps, train_s))
        print(f"Faiss IndexIVFPQ: trained in {train_s:.1f}s")
    except ImportError:
        print("faiss not installed: skipping the Faiss comparison (pip install faiss-cpu)")

    # --- print, save, plot
    print(f"\n{'index':<24}{'nprobe':>7}{'recall@10':>11}{'QPS':>10}{'train s':>9}")
    for name, nprobe, rec, qps, ts in rows:
        print(f"{name:<24}{str(nprobe):>7}{rec:>11.3f}{qps:>10.0f}{ts:>9.1f}")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "ivfpq_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "nprobe", "recall_at_10", "qps", "train_s"])
        for name, nprobe, rec, qps, ts in rows:
            w.writerow([name, nprobe, f"{rec:.4f}", f"{qps:.1f}", f"{ts:.2f}"])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed: skipping the plot")
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    styles = {
        "vectorforge_ivfpq": ("VectorForge IVF-PQ (residuals)", "o-"),
        "vectorforge_ivfpq_raw": ("VectorForge IVF-PQ (no residuals)", "^:"),
        "faiss_ivfpq": ("Faiss IndexIVFPQ", "s--"),
    }
    for key, (label, style) in styles.items():
        pts = [(r, q, p) for n, p, r, q, _ in rows if n == key]
        if not pts:
            continue
        rec, qps, probes = zip(*pts)
        ax.plot(rec, qps, style, label=label)
        if key == "vectorforge_ivfpq":
            for r, q, p in pts:
                ax.annotate(f"nprobe={p}", (r, q), textcoords="offset points", xytext=(5, 4), fontsize=7)
    flat_row = [r for r in rows if r[0] == "flat"][0]
    pq_row = [r for r in rows if r[0] == "vectorforge_pq"][0]
    ax.plot(flat_row[2], flat_row[3], "k*", markersize=12, label="Flat (exact, 512 bytes/vector)")
    ax.plot(pq_row[2], pq_row[3], "kD", markersize=7, label=f"VectorForge PQ (m={M}, no cells)")
    ax.set_yscale("log")
    ax.set_xlabel("recall@10")
    ax.set_ylabel("queries per second (log scale)")
    ax.set_title(f"IVF-PQ on SIFT10K (nlist={NLIST}, {M} bytes per vector)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "ivfpq_sweep.png", dpi=150)
    print(f"\nsaved {OUT / 'ivfpq_sweep.csv'} and {OUT / 'ivfpq_sweep.png'}")


if __name__ == "__main__":
    main()
