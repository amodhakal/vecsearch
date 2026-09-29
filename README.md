# VecSearch

Ground-up implementations of vector search in pure Python, with no dependencies beyond matplotlib for plotting

Started with brute-force KNN, then a Random Projection forest for approximate search, and now a Hierarchical Navigable Small World graph. The graph wins: it beats the forest on both axes at once.

![Recall vs throughput](docs/recall-vs-qps.png)

## How It Works

**KNN (Brute Force)** — `src/vecsearch/knn.py:5` — Scores every vector via a pluggable `Metric` (`src/vecsearch/metric.py:5`) and returns top-k with `heapq.nlargest` (`src/vecsearch/knn.py:21`). `O(n)` per query.

- `CosineSimilarityMetric` (`src/vecsearch/metric.py:10`) is a dot product on L2-normalized vectors, computed with `math.sumprod` (`src/vecsearch/metric.py:12`) rather than a Python-level loop.
- Normalization happens on load in `src/vecsearch/io.py:23-27`; zero-norm vectors are skipped at `src/vecsearch/io.py:24`.
- OOV query words raise `ValueError` (`src/vecsearch/knn.py:12`), which the CLI catches and reports without exiting (`src/vecsearch/__init__.py:23`).
- The CLI lowercases the query word (`src/vecsearch/__init__.py:19`), so `King` resolves against the lowercase GloVe vocab.

**RPForest (Approximate)** — `src/vecsearch/rpforest.py:54` — Splits the vector set with random hyperplanes until every leaf holds at most `leaf_size` words, across `n_trees` independent trees.

- **Splits are just unit normals.** The plane bisecting two randomly sampled data points is perpendicular to `a - b`, and since `|a| = |b| = 1` it passes through the origin — so a split needs no offset term, only a normal vector (`src/vecsearch/rpforest.py:99-105`). Normalizing that normal to length 1 makes every routing dot product also the signed distance to the plane, which the search reuses as its priority. This depends entirely on the loader's L2 normalization; unnormalized vectors would need a real offset.
- **Building** is a plain recursion (`src/vecsearch/rpforest.py:124`) that stops at `leaf_size` or when a plane can't separate the points, in which case it keeps an oversized leaf rather than recursing forever. Trees are drawn from a seeded RNG, so builds are reproducible.
- **Searching** explores all trees at once with a best-first heap keyed on the largest plane margin crossed to reach a node (`src/vecsearch/rpforest.py:209-237`). Following the query's own side is free; taking the far side costs that margin, so a query sitting almost on a plane gets both sides explored early — which is exactly when the split was a bad one. A monotonic tiebreak counter keeps Python from comparing node objects on equal costs.
- **Leaves produce suspects, not answers.** The tree phase only collects candidate words; re-scoring them with the real metric is the single point where the injected metric is called (`src/vecsearch/rpforest.py:239-245`). Recall therefore depends on how many candidates get rescored, which is what `search_k` controls.
- `search_k` is the recall/speed knob, analogous to Annoy's, defaulting to `k * n_trees` (`src/vecsearch/rpforest.py:199-200`).
- `add()` inserts a vector into every tree and re-splits an overfull leaf, never altering existing planes (`src/vecsearch/rpforest.py:144-179`). Cheap, but a tree built on old data won't adapt if the distribution shifts much; rebuild after large insertions.
- `find_closest` (`src/vecsearch/rpforest.py:247`) matches `KNN.find_closest`'s signature so the bench harness drives all three indexes unchanged.

**HNSW (Graph-based)** — `src/vecsearch/hnsw.py:39` — Every vector is a node in a layered proximity graph. Each node draws a random top layer and lives on layers `0..L`; layer 0 holds every node, and each layer above holds roughly `1/M` as many. Within a layer a node's links are capped at `M` (`2M` on layer 0), and that cap is what bounds a query's work — the graph itself is the index, there is nothing to scan.

- **Levels come from an exponential draw** (`src/vecsearch/hnsw.py:196`): `L = floor(-ln(1 - U) * mL)` with `mL = 1/ln M` (`src/vecsearch/hnsw.py:63`), so the expected node count per layer decays by `M` on the way up. `1 - random()` rather than `random()` because it lands in `(0, 1]`, keeping the log away from zero. A node taller than the current graph becomes the new entry point (`src/vecsearch/hnsw.py:240-242`).
- **Integer ids internally** (`src/vecsearch/hnsw.py:64-73`). Link lists hold ids, not words — cheaper to hash and store — and words are mapped back only at the API boundary.
- **Insert order is shuffled** (`src/vecsearch/hnsw.py:80-84`). GloVe is sorted by word frequency, so inserting in file order would hand the graph a systematically ordered stream. The shuffle is seeded, so builds stay reproducible.
- **Search is one beam search run per layer** (`src/vecsearch/hnsw.py:90-136`), driven by two heaps. `candidates` holds nodes still to expand and pops best-first, so it stores `(-sim, id)`; `results` holds the best `ef` seen and pops *worst*-first, since that is the one evicted when the beam overflows. The loop stops as soon as the best unexpanded candidate is worse than the beam's worst — expanding it cannot improve anything, and that single comparison is what keeps the search from wandering off.
- **Only layer 0 gets a wide beam.** Search starts at the entry node and walks down with `ef=1` (`src/vecsearch/hnsw.py:266-268`), so upper layers are just a fast greedy route to the right neighborhood; the real `ef`-wide search happens once, on the full layer 0 graph (`src/vecsearch/hnsw.py:271`).
- **Similarity, not distance.** The metric returns higher-is-closer, so "better" means a larger number throughout and every heap is inverted accordingly. `ef` is the recall/speed knob, analogous to the forest's `search_k`, and `search` raises it to at least `k` so the beam can physically hold the results (`src/vecsearch/hnsw.py:263`).
- **Neighbors are chosen for diversity, not just closeness** (`src/vecsearch/hnsw.py:142-178`). Taking the `M` closest candidates tends to pick a cluster that all link to each other and has no edge leading out of it. `_select` accepts a candidate only if it is closer to the reference node than to every neighbor already accepted, so each accepted link covers a different direction. With `keep_pruned` the rejects are used to top the list back up to `M` (`src/vecsearch/hnsw.py:172-176`), which matters most at small `M` where rejects are common.
- **Links are bidirectional and self-trimming** (`src/vecsearch/hnsw.py:226-234`). Adding the reverse edge can push an existing neighbor past its cap, so that neighbor's links are re-scored and re-selected rather than truncated blindly.
- **The two link caps are deliberately different.** A new node selects `M` neighbors on every layer including layer 0 (`src/vecsearch/hnsw.py:221`), while the *cap* applied to a neighbor's list is `2M` on layer 0 and `M` above it (`src/vecsearch/hnsw.py:219`). So layer 0 fills up past `M` through incoming reverse links and settles at a mean degree of ~26 out of a hard cap of 32. Using `2M` for the new node's own list as well is the textbook reading, and it was measured: it raises layer 0 to a flat 32, costs 2.7x more build distance computations (55M to 148M), and buys almost no recall (0.9945 to 0.9975 at `ef=10`) while *lowering* throughput, because a denser bottom layer fans out wider on every expansion. The asymmetry is the better setting at this scale.
- **Insertion runs the same descent as a query** (`src/vecsearch/hnsw.py:211-237`), beam-searching each layer it occupies at `ef_construction` width — the same `M`-link heuristic then builds its neighborhood, and the candidates found on one layer seed the search on the next. `add()` (`src/vecsearch/hnsw.py:184`) exposes this for later insertions.
- `find_closest` (`src/vecsearch/hnsw.py:283`) matches `KNN.find_closest`'s signature, passing `ef` straight through so the bench harness drives all three indexes unchanged.
- Unlike the forest, **HNSW routes through the injected metric** — there are no separate hyperplane dot products — so `CountingMetric` sees its whole cost and `plane_per_query` is legitimately 0.

## Project Structure

```
vecsearch/
├── src/vecsearch/
│   ├── knn.py      # Brute-force KNN
│   ├── rpforest.py # Random Projection forest (approximate)
│   ├── hnsw.py     # Hierarchical navigable small world graph (approximate)
│   ├── bench.py    # Recall/latency harness
│   ├── metric.py   # Metric ABC + CosineSimilarity
│   ├── io.py       # GloVe loader + L2 normalization
│   └── __init__.py # CLI entry point
├── data/
│   ├── glove.6B.100d.txt  # gitignored
│   └── truth_*.json       # Cached ground truth, committed
├── docs/
│   └── recall-vs-qps.png  # Generated plot, committed
├── results/       # Timestamped benchmark runs, committed
├── scripts/
│   └── plot_results.py    # Renders docs/ from results/
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

`.gitignore` covers only `data/glove.*`, so the downloaded vectors stay untracked while the cached ground truth, `results/` runs, and the generated plot are committed.

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

`vecsearch-bench` measures recall against exact KNN, throughput, and the work each index actually performs per query.

```bash
uv run vecsearch-bench --label baseline
# 20000 vectors
# Reusing cached ground truth: data/truth_glove.6B.100d_0.05_200_10_0.json
# knn     {}                 recall@10=1.000 qps=73.8 total/q=19999
# Results written to results/20260928_223051_baseline.jsonl
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
- `dist_per_query` — full metric evaluations per query, counted by the `CountingMetric` wrapper (`src/vecsearch/bench.py:19`) that decorates the real metric.
- `plane_per_query` — hyperplane routing dot products per query, tree indexes only. `RPForest` calls `math.sumprod` directly to route, so `CountingMetric` never sees that work; the index tallies it in `self.plane_evals` (`src/vecsearch/rpforest.py:74`) and `evaluate` resets it each sweep (`src/vecsearch/bench.py:80-81`). Zero for KNN and for HNSW, which has no separate routing step — every distance it computes goes through the metric.
- `total_per_query` — `dist + planes` (`src/vecsearch/bench.py:98`), and the number to compare indexes on. `dist_per_query` alone flatters the forest, since the plane traversal it omits is roughly half the forest's work at the cheapest setting. KNN does no routing, so its `total` equals its `dist`; so does HNSW, for the same reason.

`build_s` records index construction time separately, and it is no longer negligible: the forest builds in ~1.7s for 20K vectors, but HNSW takes 54.6s and 55.9M distance computations to build the same 20K-node graph — roughly 2,800 per inserted vector. HNSW buys its query-time win with build time, and `build_dist` in the results rows makes that trade explicit.

### Ground Truth Cache

Exact top-k per query is cached to `data/truth_<dataset>_<percent>_<queries>_<k>_<seed>.json` (`src/vecsearch/bench.py:46`). The cache is keyed on a `meta` block covering every parameter that affects the result; if any differ from the current run it is recomputed and overwritten. The first run pays full brute-force cost, later runs are instant.

### Output

One JSON object per parameter sweep is appended to `results/<timestamp>_<label>.jsonl`, so repeated runs accumulate into a shared history.

### Adding an Index

Register each index in the `runs` list at `src/vecsearch/bench.py:124-136` as a `(name, build_fn, param_sweeps)` triple. Each sweep dict is splatted into `find_closest`, which is how `search_k` is varied for the forest and `ef` for HNSW:

```python
runs = [
    ("knn", lambda: KNN(data, counter), [{}]),
    (
        "rpforest",
        lambda: RPForest(data, counter, n_trees=10, leaf_size=32),
        [{"search_k": s} for s in (100, 300, 1000, 3000)],
    ),
    (
        "hnsw",
        lambda: HNSW(data, counter, M=16, ef_construction=100),
        [{"ef": e} for e in (10, 20, 50, 100, 200)],
    ),
]
```

Every row in `results/` re-benchmarks the whole list, so an index added here shows up in all later runs, not just the one that introduced it.

An index that does work outside the injected metric should expose a `plane_evals` counter; `evaluate` detects it with `hasattr` (`src/vecsearch/bench.py:80-81`) and folds it into `total_per_query`. Without that counter the index silently under-reports its own cost.

`bench.py` carries its own `DATAPATH` at `src/vecsearch/bench.py:15`, independent of the CLI's.

## Results

GloVe 6B 100d at 5% (20,000 vectors), 200 queries, k=10, from `results/20260929_094541_hnsw.jsonl`. The forest is built with `n_trees=10`, `leaf_size=32`; HNSW with `M=16`, `ef_construction=100`.

| index | param | recall@10 | qps | dist/q | plane/q | total/q | build_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| knn | — | 1.000 | 67 | 19999 | 0 | 19999 | 0.0 |
| rpforest | `search_k=100` | 0.266 | 3787 | 109 | 105 | 215 | 1.7 |
| rpforest | `search_k=300` | 0.550 | 1817 | 309 | 138 | 447 | 1.7 |
| rpforest | `search_k=1000` | 0.800 | 650 | 1009 | 268 | 1277 | 1.7 |
| rpforest | `search_k=3000` | 0.942 | 219 | 3007 | 561 | 3567 | 1.7 |
| hnsw | `ef=10` | 0.987 | 3070 | 280 | 0 | 280 | 54.6 |
| hnsw | `ef=20` | 0.992 | 1975 | 401 | 0 | 401 | 54.6 |
| hnsw | `ef=50` | 0.995 | 1038 | 759 | 0 | 759 | 54.6 |
| hnsw | `ef=100` | 0.996 | 595 | 1268 | 0 | 1268 | 54.6 |
| hnsw | `ef=200` | 1.000 | 372 | 2150 | 0 | 2150 | 54.6 |

![Recall vs throughput](docs/recall-vs-qps.png)

Regenerate the plot after new runs land — it reads every `results/*.jsonl`, averages repeated sweeps into one point per index, and collapses the KNN rows into a single baseline:
```bash
uv run python scripts/plot_results.py
```

Each approximate index gets its own curve. Recall is on the x-axis over its full 0-1 range and throughput on a log y-axis, so the right edge is the interesting end: the forest's curve dies out at 0.94 and every HNSW point sits above and to the right of it. The HNSW curve is nearly vertical because the whole `ef` sweep lives inside the last 1.3% of recall — that compression is the result, not a plotting artifact. matplotlib is a declared dependency, used only by this script.

Reading the sweep, **compare at matched recall.** Ranking the two approximate indexes by throughput alone compares different operating points and is meaningless: the forest's fastest setting, `search_k=100` at 3824 qps, returns only 27% of the true neighbors. Holding recall at or above 0.94, the cheapest each index gets is:

| index | best setting ≥ 0.94 recall | recall | qps | work/q |
| --- | --- | --- | --- | --- |
| hnsw | `ef=10` | 0.987 | 3070 | 280 |
| rpforest | `search_k=3000` | 0.942 | 228 | 3567 |

**HNSW wins by 13x on both throughput and per-query work**, and it wins on recall too. In the plot the whole HNSW curve sits above and to the right of the forest's best point, which is the only forest setting that gets near 0.94 at all. There is no point on the forest curve that is competitive — reaching HNSW's recall by widening `search_k` would cost several times more work for a result it never reaches.

`ef` is also a much better-behaved knob. Recall saturates almost immediately — 0.987 at `ef=10`, 1.000 at `ef=200` — so 7.7x more work buys the last 1.3%. `ef=10` is the operating point, and it is only 1.3x the forest's per-query work at its cheapest setting while being 3.7x more accurate. The forest's `search_k` has the opposite shape: recall climbs all the way to the end of the sweep and never plateaus, which is what a tree partition actually looks like — recall is a function of how many candidates you rescore, and no fixed set of hyperplanes contains the neighborhoods that matter.

The cost HNSW does not hide is build time: 54.6s and 55.9M distance computations, against the forest's 1.7s and 4.0M. That is 33x the build time and 14x the build distance, in exchange for 13x less work per query than the forest's best setting (3567 to 280) and 71x less than brute force. The graph pays for itself after a few hundred queries and then keeps paying. It also dwarfs everything the forest does at query time — the entire HNSW graph costs more distance computations to build than 200 forest queries at `search_k=3000` cost to run, by a factor of 78.

The HNSW recall figures were checked against freshly computed brute-force ground truth on an independent query sample (seed 123, 200 words): 0.9945 at `ef=10`, 0.9985 at `ef=20`, 0.9995 at `ef=50`, and every node is reachable from the entry point. The query word is excluded on both sides, in `HNSW.find_closest` and in `KNN.find_closest` (`src/vecsearch/knn.py:18`), so the two are scored on the same basis.

For the forest specifically, plane traversal is 49% of its work at `search_k=100` but only 16% at `search_k=3000`. Reaching a leaf means descending all 10 trees, roughly 10 levels each, so every query pays ~100 routing dot products before it has collected any candidates at all. That fixed descent is amortized over more and more candidates as `search_k` grows, which is why cheap searches look so routing-heavy. It is also work `CountingMetric` cannot observe on its own, and the reason `total_per_query` exists.

## Dataset

[GloVe 6B 100d](https://nlp.stanford.edu/projects/glove/) — 400K vocab, 100-dim. Vectors are L2-normalized on load so dot-product == cosine similarity. Use `percent < 1.0` to subsample for faster iteration.
