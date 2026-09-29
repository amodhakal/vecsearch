# VecSearch

Ground-up implementations of vector search with pure Python and no dependencies

Started with brute-force KNN, working toward graph-based ANNS: NSW and HNSW.

## How It Works

**KNN (Brute Force)** — `src/vecsearch/knn.py:5` — Scores every vector via a pluggable `Metric` (`src/vecsearch/metric.py:5`) and returns top-k with `heapq.nlargest` (`src/vecsearch/knn.py:21`). `O(n)` per query.

- `CosineSimilarityMetric` (`src/vecsearch/metric.py:10`) is a dot product on L2-normalized vectors, computed with `math.sumprod` (`src/vecsearch/metric.py:12`) rather than a Python-level loop.
- Normalization happens on load in `src/vecsearch/io.py:23-27`; zero-norm vectors are skipped at `src/vecsearch/io.py:24`.
- OOV query words raise `ValueError` (`src/vecsearch/knn.py:12`), which the CLI catches and reports without exiting (`src/vecsearch/__init__.py:23`).
- The CLI lowercases the query word (`src/vecsearch/__init__.py:19`), so `King` resolves against the lowercase GloVe vocab.

## Project Structure

```
vecsearch/
├── src/vecsearch/
│   ├── knn.py      # Brute-force KNN
│   ├── bench.py    # Recall/latency harness
│   ├── metric.py   # Metric ABC + CosineSimilarity
│   ├── io.py       # GloVe loader + L2 normalization
│   └── __init__.py # CLI entry point
├── data/
│   ├── glove.6B.100d.txt  # gitignored
│   └── truth_*.json       # Cached ground truth, committed
├── results/       # Timestamped benchmark runs, committed
└── pyproject.toml
```

## Installation

Requires Python >=3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Download GloVe 6B 100d to `data/`:

```bash
mkdir -p data
curl -Lo data/glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip
unzip -o data/glove.6B.zip glove.6B.100d.txt -d data/
```

`.gitignore` covers only `data/glove.*`, so the downloaded vectors stay untracked while the cached ground truth and `results/` runs are committed.

## Usage

### CLI

```bash
uv run vecsearch
# Data loaded
# KNN Constructed
# How many?: 5
# Word: king
# queen
# ...
```

The CLI prints neighbor words only, without scores.

`DATAPATH` and `PERCENTAGE` are configured in `src/vecsearch/__init__.py:6-7`. The CLI loads 100% of the vocabulary by default; the benchmark defaults to 5% for faster iteration.

## Benchmark

`vecsearch-bench` measures recall against exact KNN, throughput, and distance computations per query.

```bash
uv run vecsearch-bench --label baseline
# 20000 vectors
# Reusing cached ground truth: data/truth_glove.6B.100d_0.05_200_10_0.json
# knn    {}                recall@10=1.000 qps=<varies> dist/q=19999
# Results written to results/20260928_222612_baseline.jsonl
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--percent` | `0.05` | Fraction of vocabulary to load |
| `--queries` | `200` | Number of random query words |
| `--k` | `10` | Neighbors per query |
| `--seed` | `0` | Seeds query sampling |
| `--label` | `run` | Label embedded in the output filename |

### Metrics

- `recall@k` — overlap with the exact top-k, where brute-force KNN is the ground truth.
- `qps` — queries per second.
- `dist_per_query` — distance computations per query, counted by the `CountingMetric` wrapper (`src/vecsearch/bench.py:18`) that decorates the real metric. This is the number that separates a graph index from brute force: it drops toward the neighbor count instead of tracking `N`.

`build_s` records index construction time separately, which matters for NSW and HNSW where build is the expensive phase.

### Ground Truth Cache

Exact top-k per query is cached to `data/truth_<dataset>_<percent>_<queries>_<k>_<seed>.json` (`src/vecsearch/bench.py:30`). The cache is keyed on a `meta` block covering every parameter that affects the result; if any differ from the current run it is recomputed and overwritten. The first run pays full brute-force cost, later runs are instant.

### Output

One JSON object per parameter sweep is appended to `results/<timestamp>_<label>.jsonl`, so repeated runs accumulate into a shared history.

### Adding an Index

Register each index in the `runs` list at `src/vecsearch/bench.py:114-118` as a `(name, build_fn, param_sweeps)` triple. Each sweep dict is splatted into `find_closest`, which is how `ef` will be varied once NSW lands:

```python
runs = [
    ("knn", lambda: KNN(data, counter), [{}]),
    ("nsw", lambda: NSW(data, counter, M=16), [{"ef": e} for e in (10, 20, 50, 100)]),
]
```

`bench.py` carries its own `DATAPATH` at `src/vecsearch/bench.py:13`, independent of the CLI's.

## Dataset

[GloVe 6B 100d](https://nlp.stanford.edu/projects/glove/) — 400K vocab, 100-dim. Vectors are L2-normalized on load so dot-product == cosine similarity. Use `percent < 1.0` to subsample for faster iteration.
