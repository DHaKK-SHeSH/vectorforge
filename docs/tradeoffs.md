# The trade-off that mattered most

**Decision:** for a million-vector collection where memory is limited, should the index be **HNSW** or **IVF-PQ with exact re-ranking**?

**Answer from the measurements:** if the full vectors fit in RAM, use HNSW: it is the fastest at high recall. If they don't, use IVF-PQ + re-ranking: about 20-45x less RAM for similar recall, paid for with one small disk read per query.

## Why this is the decision that matters

Speed alone points to HNSW, and recall alone points to exact search. But at real scale the first limit you hit is usually **memory**: SIFT1M's raw vectors take 512 MB, a billion of them take 512 GB. HNSW keeps every full vector in RAM *plus* its links, so it gets *bigger* than the data. IVF-PQ keeps only 8-byte codes, so it is 64x smaller, but on its own it is too inaccurate.

## The measurements

| | HNSW (M = 16) | IVF-PQ alone | IVF-PQ + re-ranking (k_factor 10) |
|---|---|---|---|
| RAM, SIFT10K | about 7.8 MB (vectors + links) | 0.34 MB | 0.34 MB (+ 5.12 MB read from disk as needed) |
| RAM, SIFT1M | 777 MB | 16.7 MB | about 17 MB (+ 512 MB on disk) |
| recall@10, SIFT10K | 0.992 at ef_search = 64 | 0.607 | 0.952 (nprobe 8), 0.980 (nprobe 16) |
| recall@10, SIFT1M | 0.920 at ef_search = 128 | 0.370 | not measured on 1M yet |
| build, SIFT1M (my Python) | 58 min | 3.6 min | 3.6 min |

Sources: `docs/hnsw_sweep.csv`, `docs/ivfpq_sweep.csv`, `docs/refine_sweep.csv`, `docs/sift1m_results.csv`.

## What the numbers say

1. **Re-ranking turns IVF-PQ from unusable into competitive.** 0.61 → 0.95 recall on SIFT10K at the same speed, because measuring 100 exact distances per query is cheap compared with the search itself.
2. **The re-ranked index can only be as good as its shortlist.** At nprobe = 8 recall stopped at 0.961, IVF's own recall at nprobe = 8: re-ranking fixes the *order*, not *which cells were searched*. More cells (nprobe 16) raised the ceiling to 0.99.
3. **HNSW still wins on pure speed per query** when everything fits in RAM, and the heuristic (v1.1) made it better with *fewer* links.
4. **Build time matters too:** in Python, HNSW took 58 minutes on 1M vectors, IVF-PQ under 4.

## Decision rule

- Vectors fit comfortably in RAM and queries must be fast: **HNSW** (with the heuristic).
- RAM is the limit: **IVF-PQ + re-ranking**, full vectors on SSD, tune nprobe for the recall ceiling and k_factor for the order.
- Need 99%+ on a small collection: **exact search**.

## Caveats

- My indexes are Python + NumPy. Faiss runs the same algorithms 4-120x faster; the *ratios* between my indexes are what I trust, not their absolute speeds.
- Some sweeps ran on my laptop, some in a cloud container, and SIFT1M on Kaggle: compare speeds only within one table.
- Re-ranking wasn't run on SIFT1M. The next step would be to repeat table row 4 there, with the full vectors in a memory-mapped file.
