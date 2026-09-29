import argparse
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path

from vecsearch.io import load_normalized_data
from vecsearch.knn import KNN
from vecsearch.metric import CosineSimilarityMetric, Metric
from vecsearch.rpforest import RPForest

DATAPATH = "data/glove.6B.100d.txt"
CACHE = Path("data")
RESULTS = Path("results")


class CountingMetric(Metric):
    """Wraps any Metric and counts how many distances get computed."""

    def __init__(self, inner: Metric) -> None:
        self.inner = inner
        self.calls = 0

    def __call__(self, vec1, vec2):
        self.calls += 1
        return self.inner(vec1, vec2)


def get_truth(data, percent, n_queries, k, seed):
    """Exact top-k per query from brute-force KNN, cached on disk.

    The cache is reused only if every parameter that affects the truth
    matches; otherwise it is recomputed and overwritten.
    """
    CACHE.mkdir(exist_ok=True)
    meta = {
        "dataset": Path(DATAPATH).name,
        "percent": percent,
        "n_vectors": len(data),
        "n_queries": n_queries,
        "k": k,
        "seed": seed,
    }
    path = CACHE / (
        f"truth_{Path(DATAPATH).stem}_{percent}_{n_queries}_{k}_{seed}.json"
    )

    if path.exists():
        z = json.loads(path.read_text())
        if z.get("meta") == meta:
            print(f"Reusing cached ground truth: {path}")
            return z["queries"], {q: set(v) for q, v in z["truth"].items()}
        print(f"Cached truth at {path} doesn't match current parameters, recomputing")

    rng = random.Random(seed)
    queries = rng.sample(list(data), n_queries)
    exact = KNN(data, CosineSimilarityMetric())
    truth = {}
    for i, q in enumerate(queries):
        truth[q] = {w for w, _ in exact.find_closest(q, k)}
        if i % 50 == 0:
            print(f"  ground truth {i}/{n_queries}")

    path.write_text(
        json.dumps(
            {
                "meta": meta,
                "queries": queries,
                "truth": {q: sorted(v) for q, v in truth.items()},
            }
        )
    )
    return queries, truth


def evaluate(index, counter, queries, truth, k, **params):
    counter.calls = 0
    if hasattr(index, "plane_evals"):  # only tree indexes have this
        index.plane_evals = 0

    hits = 0
    t0 = time.perf_counter()
    for q in queries:
        found = {w for w, _ in index.find_closest(q, k, **params)}
        hits += len(found & truth[q])
    elapsed = time.perf_counter() - t0

    n = len(queries)
    dist = counter.calls / n
    planes = getattr(index, "plane_evals", 0) / n
    return {
        "recall": hits / (n * k),
        "qps": n / elapsed,
        "dist_per_query": dist,
        "plane_per_query": planes,
        "total_per_query": dist + planes,  # compare indexes on this one
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percent", type=float, default=0.05)  # 5% = 20k vectors
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    started = datetime.now()
    RESULTS.mkdir(exist_ok=True)
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.label)
    log_path = RESULTS / f"{started:%Y%m%d_%H%M%S}_{safe_label}.jsonl"

    data = load_normalized_data(DATAPATH, args.percent)
    print(f"{len(data)} vectors")
    queries, truth = get_truth(data, args.percent, args.queries, args.k, args.seed)

    counter = CountingMetric(CosineSimilarityMetric())

    # Register indexes here: (name, build_fn, list_of_search_param_dicts)
    runs = [
        ("knn", lambda: KNN(data, counter), [{}]),
        (
            "rpforest",
            lambda: RPForest(data, counter, n_trees=10, leaf_size=32),
            [{"search_k": s} for s in (100, 300, 1000, 3000)],
        ),
        # ("nsw", lambda: NSW(data, counter, M=16), [{"ef": e} for e in (10, 20, 50, 100)]),
    ]

    for name, build, sweeps in runs:
        t0 = time.perf_counter()
        index = build()
        build_s = time.perf_counter() - t0
        for params in sweeps:
            r = evaluate(index, counter, queries, truth, args.k, **params)
            row = {
                "timestamp": started.isoformat(timespec="seconds"),
                "label": args.label,
                "index": name,
                "params": params,
                "N": len(data),
                "k": args.k,
                "seed": args.seed,
                "build_s": round(build_s, 2),
                **{a: round(b, 4) for a, b in r.items()},
            }
            with log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(
                f"{name:8} {str(params):18} recall@{args.k}={r['recall']:.3f} "
                f"qps={r['qps']:.1f} total/q={r['total_per_query']:.0f}"
            )

    print(f"Results written to {log_path}")


if __name__ == "__main__":
    main()
