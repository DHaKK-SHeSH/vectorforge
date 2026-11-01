"""v1.1 extra B: HNSW with the neighbour-selection heuristic vs plain "closest M" (SIFT10K).

For M = 8 and 16 (ef_construction = 100) it builds a graph with heuristic=False and
heuristic=True and measures recall@10 and QPS for ef_search = 10 ... 128, plus build
time and links per vector. Faiss IndexHNSWFlat (which uses the heuristic) is the reference.

Run from the project root:
    python scripts/sweep_hnsw_heuristic.py

Outputs:
    docs/hnsw_heuristic_sweep.csv   the numbers
    docs/hnsw_heuristic_sweep.png   recall vs QPS
"""

import csv
import time
from pathlib import Path

from vectorforge.hnsw import HNSWIndex
from vectorforge.io import load_sift
from vectorforge.metrics import recall_at_k

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "siftsmall"
OUT = ROOT / "docs"
MS = [8, 16]
EF_CONSTRUCTION = 100
EF_SEARCHES = [10, 16, 32, 64, 128]
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
    d = xb.shape[1]
    rows = []   # (index, M, ef_search, recall, qps, build_s, links_per_vector)

    for m in MS:
        for heuristic, name in ((False, "vectorforge_plain"), (True, "vectorforge_heuristic")):
            index = HNSWIndex(M=m, ef_construction=EF_CONSTRUCTION, heuristic=heuristic)
            t0 = time.perf_counter()
            index.add(xb)
            build_s = time.perf_counter() - t0
            links = sum(len(v) for layer in index.graph for v in layer.values()) / len(xb)
            print(f"{name} M={m}: built in {build_s:.1f}s, {links:.1f} links/vector")
            for ef in EF_SEARCHES:
                index.ef_search = ef
                ids, qps = timed_search(index, xq, K)
                rows.append((name, m, ef, recall_at_k(ids, gt, K), qps, build_s, links))

    try:
        import faiss
        faiss.omp_set_num_threads(1)
        for m in MS:
            findex = faiss.IndexHNSWFlat(d, m)
            findex.hnsw.efConstruction = EF_CONSTRUCTION
            t0 = time.perf_counter()
            findex.add(xb)
            build_s = time.perf_counter() - t0
            for ef in EF_SEARCHES:
                findex.hnsw.efSearch = ef
                ids, qps = timed_search(findex, xq, K)
                rows.append(("faiss_hnsw", m, ef, recall_at_k(ids, gt, K), qps, build_s, float("nan")))
    except ImportError:
        print("faiss not installed: skipping the Faiss reference (pip install faiss-cpu)")

    print(f"\n{'index':<24}{'M':>4}{'ef':>6}{'recall@10':>11}{'QPS':>9}{'build s':>9}{'links':>7}")
    for name, m, ef, rec, qps, bs, links in rows:
        print(f"{name:<24}{m:>4}{ef:>6}{rec:>11.3f}{qps:>9.0f}{bs:>9.1f}{links:>7.1f}")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "hnsw_heuristic_sweep.csv", "w", newline="") as f:
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

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    styles = {"vectorforge_plain": ("VectorForge, closest M (v1.0)", "o:"),
              "vectorforge_heuristic": ("VectorForge, heuristic (v1.1)", "o-"),
              "faiss_hnsw": ("Faiss (heuristic)", "s--")}
    for ax, m in zip(axes, MS):
        for key, (label, style) in styles.items():
            pts = [(r, q) for n, mm, _, r, q, _, _ in rows if n == key and mm == m]
            if pts:
                rec, qps = zip(*pts)
                ax.plot(rec, qps, style, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("recall@10")
        ax.set_title(f"M = {m}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("queries per second (log scale)")
    fig.suptitle(f"HNSW neighbour selection on SIFT10K (ef_construction={EF_CONSTRUCTION}, ef_search 10 to 128)")
    fig.tight_layout()
    fig.savefig(OUT / "hnsw_heuristic_sweep.png", dpi=150)
    print(f"\nsaved {OUT / 'hnsw_heuristic_sweep.csv'} and {OUT / 'hnsw_heuristic_sweep.png'}")


if __name__ == "__main__":
    main()
