# VectorForge

Approximate nearest-neighbour search written from scratch in NumPy — exact search, IVF, product quantisation and HNSW — benchmarked against Faiss on recall, speed and memory.

> 🚧 In progress: exact search, IVF and product quantisation are done. IVF-PQ and HNSW are next.

## Results

SIFT10K (10,000 base vectors, 100 queries, 128 dimensions), k = 10, measured on my laptop. Faiss runs single-threaded for a fair comparison. Speeds come from separate runs, and laptop timings vary by up to about 1.5× between runs (Flat measured 4,568 and 7,657 QPS), so compare speeds within one sweep.

| Index | Settings | recall@10 | queries/s | memory (MB) |
|---|---|---|---|---|
| Flat (exact) | — | 1.000 | 4,568 | 5.16 |
| IVF (VectorForge) | nlist=100, nprobe=8 | 0.961 | 1,920 | ≈5.25 |
| IVF (VectorForge) | nlist=100, nprobe=1 | 0.501 | 21,406 | ≈5.25 |
| IVF (Faiss) | nlist=100, nprobe=8 | 0.922 | 32,438 | — |
| PQ (VectorForge) | m=8, 8 bits (8 bytes/vector) | 0.558 | 2,962 | 0.08 + 0.13 codebooks |
| PQ (VectorForge) | m=32, 8 bits (32 bytes/vector) | 0.833 | 752 | 0.32 + 0.13 codebooks |
| PQ (Faiss) | m=8, 8 bits | 0.586 | 14,243 | 0.08 + 0.13 codebooks |
| IVF-PQ | — | — | — | — |
| HNSW | — | — | — | — |

## IVF: speed vs recall

![Recall vs queries per second for VectorForge IVF, Faiss IVF and exact search](docs/ivf_sweep.png)

Full numbers for every nprobe: [`docs/ivf_sweep.csv`](docs/ivf_sweep.csv).

**Findings**

- **Recall rises with nprobe, with strongly diminishing returns.** A query near the border of its cell has some of its true neighbours in the neighbouring cells, and nprobe = 1 never looks there (recall 0.50). Probing 8 of 100 cells — about 8% of the data — already finds 96% of the true neighbours; reaching 100% takes nprobe = 32, four times the work for the last 4%.
- **On 10,000 vectors my IVF is slower than exact search from nprobe = 8 on, even though it does about 10× less arithmetic.** Exact search computes all 100 × 10,000 distances in a single matrix multiply in optimised compiled code. My IVF search loops over the queries in Python, and for each one gathers and copies the candidate vectors. At this size that per-query overhead outweighs the work saved. Exact search's cost grows linearly with the database while IVF's per-query cost stays small, so the picture should flip on SIFT1M (next benchmark).
- **Faiss is roughly 7–22× faster with the same algorithm.** Its search loop runs in compiled C++ with no per-query interpreter overhead, it stores each cell's vectors contiguously in memory instead of gathering them on every query, and it uses SIMD instructions to process several numbers at once.
- **At the same nprobe my recall is about the same as Faiss's at nprobe = 1 and higher from nprobe = 2 to 16** (e.g. 0.961 vs 0.922 at nprobe = 8). Both are IVF; the difference comes from k-means: I initialise with k-means++, while Faiss starts from randomly sampled centroids by default, so the two indexes end up with different cells and different vectors near cell borders. Both reach recall 1.0 once enough cells are probed, which confirms the implementation is correct.

## Product quantisation: memory vs recall

Each 128-dimensional vector is cut into m pieces. Each piece is replaced by the number of its nearest entry in a 256-entry codebook learned with k-means on that piece, so a vector is stored as m bytes instead of 512. Search never decodes: per query it builds one small distance table per piece, then a vector's distance is the sum of m table lookups (asymmetric distance computation). A test checks that this equals the exact distance to the decoded vector.

![Recall vs bytes per vector, and speed vs recall, for VectorForge PQ, Faiss IndexPQ and exact search](docs/pq_sweep.png)

| m (bytes/vector) | compression | VectorForge recall@10 | VectorForge QPS | Faiss recall@10 | Faiss QPS | train time (mine / Faiss) |
|---|---|---|---|---|---|---|
| 4 | 128× | 0.396 | 5,023 | 0.408 | 22,840 | 13.6 s / 1.6 s |
| 8 | 64× | 0.558 | 2,962 | 0.586 | 14,243 | 18.0 s / 2.7 s |
| 16 | 32× | 0.719 | 1,600 | 0.721 | 5,954 | 27.0 s / 5.8 s |
| 32 | 16× | 0.833 | 752 | 0.841 | 3,934 | 47.3 s / 25.5 s |
| Flat (exact) | 1× (512 bytes) | 1.000 | 7,657 | — | — | — |

Full numbers: [`docs/pq_sweep.csv`](docs/pq_sweep.csv).

**Findings**

- **More pieces give better recall, paid for in memory.** Going from 4 to 32 bytes per vector raises recall@10 from 0.40 to 0.83. With more pieces each piece is shorter (32 numbers at m = 4, 4 numbers at m = 32), so 256 codebook entries cover it much more densely. The rebuilt vector is a finer approximation of the original, and the ranking gets closer to the exact one.
- **PQ trades recall for a lot of memory.** At m = 8 a vector takes 8 bytes instead of 512 (64× smaller) for recall 0.56. At m = 32 it's 16× smaller for 0.83. On 10,000 vectors that saving doesn't matter, but on a billion SIFT vectors it's 8 GB instead of 512 GB, the difference between fitting in RAM or not. On its own the recall is too low for most uses. In practice PQ is combined with IVF (next) and/or a re-ranking step: take a short candidate list from PQ, then re-rank it with exact distances.
- **On this dataset my PQ is slower than exact search, and halves in speed every time m doubles.** It's not the arithmetic: per stored vector PQ does m lookups instead of 128 multiply-adds, and the distance tables are tiny (100 queries × 256 entries per piece). The cost is that every piece's lookup (`table[:, codes[:, j]]`) builds a new 100 × 10,000 array, so the search time is m full passes over memory (5,023 → 2,962 → 1,600 → 752 QPS). Exact search is a single optimised matrix multiply.
- **Faiss gets the same recall (within 0.03) but searches 3.7–5.2× faster and trains 2–9× faster.** Same algorithm; the small recall differences come from k-means initialisation and training settings. Faiss's compiled C++ loop walks over the codes once and adds up the m table values for each vector. The tables (m × 256 floats) stay in the CPU cache, and no large temporary arrays are created.

## Run it

```bash
pip install -e ".[dev]"
pytest                          # unit tests
python -m vectorforge.bench     # exact-search benchmark on SIFT10K (see data/README.md to download it)
pip install faiss-cpu
python scripts/sweep_ivf.py     # IVF nprobe sweep vs Faiss → docs/ivf_sweep.csv and .png
python scripts/sweep_pq.py      # PQ sweep over m vs Faiss → docs/pq_sweep.csv and .png
```

## Layout

- `vectorforge/io.py` — readers for the `.fvecs` / `.ivecs` dataset format
- `vectorforge/flat.py` — exact search (the baseline)
- `vectorforge/metrics.py` — recall@k
- `vectorforge/kmeans.py` — k-means with k-means++ initialisation
- `vectorforge/ivf.py` — IVF index: k-means cells, inverted lists, nprobe search
- `vectorforge/pq.py` — product quantisation: per-piece codebooks, 1-byte codes, lookup-table (ADC) search
- `vectorforge/bench.py` — one harness that measures every index the same way
- `scripts/sweep_ivf.py` — IVF recall/speed sweep against Faiss
- `scripts/sweep_pq.py` — PQ recall/memory/speed sweep against Faiss
