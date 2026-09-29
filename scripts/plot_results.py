"""Render the recall/throughput tradeoff from results/*.jsonl.

matplotlib is not a project dependency. Run it ephemerally:

    uv run --with matplotlib python scripts/plot_results.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("results")
OUT = Path("docs/recall-vs-qps.png")


def load_rows():
    rows = []
    for path in sorted(RESULTS.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    rows = load_rows()

    # KNN is exact, so every one of its rows is the same point: recall 1.0 at
    # roughly the same qps. Average them into a single baseline.
    exact = [r["qps"] for r in rows if r["index"] == "knn"]
    knn_qps = sum(exact) / len(exact)

    # Approximate indexes get one point per parameter sweep, ordered by qps so
    # the curve is drawn left-to-right.
    approx = [r for r in rows if r["index"] != "knn"]
    approx.sort(key=lambda r: -r["qps"])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        [r["qps"] for r in approx],
        [r["recall"] for r in approx],
        marker="o",
        color="#2b6cb0",
        linewidth=1.5,
        label="RPForest (approx)",
        zorder=2,
    )
    for r in approx:
        ax.annotate(
            f"search_k={r['params']['search_k']}",
            (r["qps"], r["recall"]),
            textcoords="offset points",
            xytext=(-8, 10),
            ha="right",
            fontsize=8,
            color="#2b6cb0",
        )

    ax.scatter(
        [knn_qps],
        [1.0],
        marker="*",
        s=220,
        color="#c05621",
        zorder=3,
        label="KNN (exact)",
    )
    ax.annotate(
        f"KNN {knn_qps:.0f} qps",
        (knn_qps, 1.0),
        textcoords="offset points",
        xytext=(12, -4),
        fontsize=8,
        color="#c05621",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Throughput (queries per second, log scale)")
    ax.set_ylabel("recall@10 vs exact KNN")
    ax.set_title("Recall vs throughput — GloVe 6B 100d, 20K vectors, 200 queries")
    ax.set_ylim(0, 1.05)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
