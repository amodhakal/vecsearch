# VecSearch

Ground-up implementations of vector search with pure Python and no depdencies

Started with brute-force KNN, working toward graph-based ANNS: NSW and HNSW.

## How It Works

**KNN (Brute Force)** — `src/vecsearch/knn.py:5` — Scores every vector via a pluggable `Metric` (`src/vecsearch/metric.py:4`) and returns top-k with `heapq.nlargest`. `O(n)` per query.

- `CosineSimilarityMetric` is a dot product on L2-normalized vectors. Normalization happens on load in `src/vecsearch/io.py:22`.
- OOV query words raise `ValueError`.

## Project Structure

```
vecsearch/
├── src/vecsearch/
│   ├── knn.py      # Brute-force KNN
│   ├── metric.py   # Metric ABC + CosineSimilarity
│   ├── io.py       # GloVe loader + L2 normalization
│   └── __init__.py # CLI entry point
├── data/glove.6B.100d.txt
└── pyproject.toml
```

## Installation

Requires Python >=3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Download GloVe 6B 100d to `data/`:

```bash
curl -Lo data/glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip
unzip -o data/glove.6B.zip glove.6B.100d.txt -d data/
```

## Usage

### CLI

```bash
uv run vecsearch
# Data loaded
# KNN Constructed
# How many?: 5
# Word: king
# ('queen', 0.78)
# ...
```

`DATAPATH` and `PERCENTAGE` are configured in `src/vecsearch/__init__.py:6`.

## Dataset

[GloVe 6B 100d](https://nlp.stanford.edu/projects/glove/) — 400K vocab, 100-dim. Vectors are L2-normalized on load so dot-product == cosine similarity. Use `percent < 1.0` to subsample for faster iteration.

## Roadmap

- [x] **KNN** — Brute-force exact search
- [ ] **NSW** — Navigable Small World graph (Malkov et al. 2014)
- [ ] **HNSW** — Hierarchical NSW (Malkov & Yashunin 2016)
