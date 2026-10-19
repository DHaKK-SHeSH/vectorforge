"""Session 4: measure the IVF speed/recall trade-off on SIFT10K.

For each nprobe it measures recall@10 and queries per second (QPS) for
  - your IVFIndex (VectorForge)
  - Faiss's IndexIVFFlat with the same nlist / nprobe (if faiss is installed)
and the exact FlatIndex as the reference point.

Run from the project root:
    python scripts/sweep_ivf.py

Outputs:
    docs/ivf_sweep.csv   the numbers
    docs/ivf_sweep.png   the recall-vs-QPS plot for the README
"""

import csv
import time
from pathlib import Path

import numpy as np

from vectorforge.flat import FlatIndex
from vectorforge.io import load_sift, read_fvecs
from vectorforge.ivf import IVFIndex
from vectorforge.metrics import recall_at_k

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "siftsmall"
OUT = ROOT / "docs"
NLIST = 100
NPROBES = [1, 2, 4, 8, 16, 32, 64, 100]
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
    print(f"base {xb.shape}  queries {xq.shape}  train {xt.shape}  nlist={NLIST}\n")
    rows = []

    # --- exact baseline
    flat = FlatIndex("l2")
    flat.add(xb)
    ids, qps = timed_search(flat, xq, K)
    rows.append(("flat", "-", recall_at_k(ids, gt, K), qps))

    # --- your IVF
    ivf = IVFIndex(nlist=NLIST)
    t0 = time.perf_counter()
    ivf.train(xt)
    train_s = time.perf_counter() - t0
    ivf.add(xb)
    sizes = np.array([len(lst) for lst in ivf.lists])
    print(f"VectorForge IVF: trained in {train_s:.1f}s; list sizes min/mean/max = "
          f"{sizes.min()}/{sizes.mean():.0f}/{sizes.max()}")
    for nprobe in NPROBES:
        ivf.nprobe = nprobe
        ids, qps = timed_search(ivf, xq, K)
        rows.append(("vectorforge_ivf", nprobe, recall_at_k(ids, gt, K), qps))

    # --- Faiss IVF, same settings (single thread, so the comparison is fair)
    try:
        import faiss
        faiss.omp_set_num_threads(1)
        quantizer = faiss.IndexFlatL2(xb.shape[1])
        fivf = faiss.IndexIVFFlat(quantizer, xb.shape[1], NLIST)
        fivf.train(xt)
        fivf.add(xb)
        for nprobe in NPROBES:
            fivf.nprobe = nprobe
            ids, qps = timed_search(fivf, xq, K)
            rows.append(("faiss_ivf", nprobe, recall_at_k(ids, gt, K), qps))
    except ImportError:
        print("faiss not installed: skipping the Faiss comparison (pip install faiss-cpu)")

    # --- print, save, plot
    print(f"\n{'index':<17}{'nprobe':>7}{'recall@10':>11}{'QPS':>10}")
    for name, nprobe, rec, qps in rows:
        print(f"{name:<17}{str(nprobe):>7}{rec:>11.3f}{qps:>10.0f}")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "ivf_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "nprobe", "recall_at_10", "qps"])
        for name, nprobe, rec, qps in rows:
            w.writerow([name, nprobe, f"{rec:.4f}", f"{qps:.1f}"])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed: skipping the plot")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = {"vectorforge_ivf": ("VectorForge IVF", "o-"), "faiss_ivf": ("Faiss IVFFlat", "s--")}
    for key, (label, style) in styles.items():
        pts = [(r, q, p) for n, p, r, q in rows if n == key]
        if not pts:
            continue
        rec, qps, probes = zip(*pts)
        ax.plot(rec, qps, style, label=label)
        if key == "vectorforge_ivf":
            for r, q, p in pts:
                ax.annotate(f"nprobe={p}", (r, q), textcoords="offset points", xytext=(5, 4), fontsize=7)
    flat_row = [r for r in rows if r[0] == "flat"][0]
    ax.plot(flat_row[2], flat_row[3], "k*", markersize=12, label="Flat (exact)")
    ax.set_yscale("log")
    ax.set_xlabel("recall@10")
    ax.set_ylabel("queries per second (log scale)")
    ax.set_title(f"IVF on SIFT10K (nlist={NLIST}): speed vs recall")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "ivf_sweep.png", dpi=150)
    print(f"\nsaved {OUT / 'ivf_sweep.csv'} and {OUT / 'ivf_sweep.png'}")


if __name__ == "__main__":
    main()
