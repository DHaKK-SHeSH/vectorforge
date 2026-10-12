# VectorForge

Approximate nearest-neighbour search written from scratch in NumPy — exact search, IVF, product quantisation and HNSW — benchmarked against Faiss on recall, speed and memory.

> 🚧 In progress. Results land here as each index is finished.

## Results

| Index | recall@10 | queries/s | memory (MB) | build (s) |
|---|---|---|---|---|
| Flat (exact) | — | — | — | — |
| IVF | — | — | — | — |
| IVF-PQ | — | — | — | — |
| HNSW | — | — | — | — |
| Faiss (reference) | — | — | — | — |

## Run it

```bash
pip install -e ".[dev]"
pytest                      # unit tests
python -m vectorforge.bench # benchmark on SIFT10K (see data/README.md to download it)
```

## Layout

- `vectorforge/io.py` — readers for the `.fvecs` / `.ivecs` dataset format
- `vectorforge/flat.py` — exact search (the baseline)
- `vectorforge/metrics.py` — recall@k
- `vectorforge/bench.py` — one harness that measures every index the same way
