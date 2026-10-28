"""HNSW: speed vs recall over ef_search, for two graph sizes M (SIFT10K).

For M = 8 and M = 16 (ef_construction = 100) it builds a graph and, for each
ef_search, measures recall@10 and queries per second (QPS) for
  - your HNSWIndex (VectorForge)
  - Faiss's IndexHNSWFlat with the same M / efConstruction / efSearch (if faiss is installed)
plus build time and links per vector, with exact Flat search as the reference point.

Run from the project root:
    python scripts/sweep_hnsw.py

Building the graph is the slow part (two graphs of 10,000 vectors, one insert at
a time in Python): expect a few minutes.

Outputs:
    docs/hnsw_sweep.csv   the numbers
    docs/hnsw_sweep.png   recall vs QPS
"""

import csv
import time
from pathlib import Path

from vectorforge.flat import FlatIndex
from vectorforge.hnsw import HNSWIndex
from vectorforge.io import load_sift
from vectorforge.metrics import recall_at_k

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "siftsmall"
OUT = ROOT / "docs"
MS = [8, 16]
EF_CONSTRUCTION = 100
EF_SEARCHES = [10, 16, 32, 64, 128, 256]
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
    d = xb.shape[1]
    print(f"base {xb.shape}  queries {xq.shape}  ef_construction={EF_CONSTRUCTION}\n")
    rows = []   # (index, M, ef_search, recall, qps, build_s, links_per_vector)

    # --- exact baseline
    flat = FlatIndex("l2")
    flat.add(xb)
    ids, qps = timed_search(flat, xq, K)
    rows.append(("flat", "-", "-", recall_at_k(ids, gt, K), qps, 0.0, 0.0))

    # --- your HNSW
    for m in MS:
        index = HNSWIndex(M=m, ef_construction=EF_CONSTRUCTION)
        t0 = time.perf_counter()
        index.add(xb)
        build_s = time.perf_counter() - t0
        links = sum(len(nbrs) for layer in index.graph for nbrs in layer.values()) / len(xb)
        sizes = [len(layer) for layer in index.graph]
        print(f"VectorForge HNSW M={m}: built in {build_s:.1f}s, {links:.1f} links/vector, "
              f"vectors per layer {sizes}")
        for ef in EF_SEARCHES:
            index.ef_search = ef
            ids, qps = timed_search(index, xq, K)
            rows.append(("vectorforge_hnsw", m, ef, recall_at_k(ids, gt, K), qps, build_s, links))

    # --- Faiss HNSW, same settings (single thread, so the comparison is fair)
    try:
        import faiss
        faiss.omp_set_num_threads(1)
        for m in MS:
            findex = faiss.IndexHNSWFlat(d, m)
            findex.hnsw.efConstruction = EF_CONSTRUCTION
            t0 = time.perf_counter()
            findex.add(xb)
            build_s = time.perf_counter() - t0
            print(f"Faiss IndexHNSWFlat M={m}: built in {build_s:.1f}s")
            for ef in EF_SEARCHES:
                findex.hnsw.efSearch = ef
                ids, qps = timed_search(findex, xq, K)
                rows.append(("faiss_hnsw", m, ef, recall_at_k(ids, gt, K), qps, build_s, float("nan")))
    except ImportError:
        print("faiss not installed: skipping the Faiss comparison (pip install faiss-cpu)")

    # --- print, save, plot
    print(f"\n{'index':<18}{'M':>4}{'ef':>6}{'recall@10':>11}{'QPS':>10}{'build s':>9}")
    for name, m, ef, rec, qps, bs, _ in rows:
        print(f"{name:<18}{str(m):>4}{str(ef):>6}{rec:>11.3f}{qps:>10.0f}{bs:>9.1f}")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "hnsw_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "M", "ef_search", "recall_at_10", "qps", "build_s", "links_per_vector"])
        for name, m, ef, rec, qps, bs, links in rows:
            w.writerow([name, m, ef, f"{rec:.4f}", f"{qps:.1f}", f"{bs:.2f}", f"{links:.1f}"])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed: skipping the plot")
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    styles = {("vectorforge_hnsw", 8): ("VectorForge HNSW, M=8", "^:"),
              ("vectorforge_hnsw", 16): ("VectorForge HNSW, M=16", "o-"),
              ("faiss_hnsw", 8): ("Faiss HNSW, M=8", "v:"),
              ("faiss_hnsw", 16): ("Faiss HNSW, M=16", "s--")}
    for (key, m), (label, style) in styles.items():
        pts = [(r, q, e) for n, mm, e, r, q, _, _ in rows if n == key and mm == m]
        if not pts:
            continue
        rec, qps, efs = zip(*pts)
        ax.plot(rec, qps, style, label=label)
        if key == "vectorforge_hnsw" and m == 16:
            for r, q, e in pts:
                ax.annotate(f"ef={e}", (r, q), textcoords="offset points", xytext=(-38, -4), fontsize=7)
    flat_row = [r for r in rows if r[0] == "flat"][0]
    ax.plot(flat_row[3], flat_row[4], "k*", markersize=12, label="Flat (exact)")
    ax.set_yscale("log")
    ax.set_xlabel("recall@10")
    ax.set_ylabel("queries per second (log scale)")
    ax.set_title(f"HNSW on SIFT10K (ef_construction={EF_CONSTRUCTION}): speed vs recall")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "hnsw_sweep.png", dpi=150)
    print(f"\nsaved {OUT / 'hnsw_sweep.csv'} and {OUT / 'hnsw_sweep.png'}")


if __name__ == "__main__":
    main()
