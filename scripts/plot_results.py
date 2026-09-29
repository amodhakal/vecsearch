"""Render the recall/throughput tradeoff from results/*.jsonl.

uv run python scripts/plot_results.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("results")
OUT = Path("docs/recall-vs-qps.png")

# One curve per approximate index, cycled in a fixed order so colors stay
# stable as new indexes are added.
COLORS = ["#2b6cb0", "#2f855a", "#b7791f", "#805ad5"]
KNN_COLOR = "#c05621"


def label_params(params: dict) -> str:
    """Render a sweep's search params generically: {"ef": 10} -> "ef=10"."""
    return ", ".join(f"{k}={v}" for k, v in params.items())


def load_rows():
    rows = []
    for path in sorted(RESULTS.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_series(rows) -> tuple[list[dict], float]:
    """Collapse the raw rows into per-index curves plus the KNN baseline.

    Every results file re-benchmarks the whole suite, so the same sweep shows up
    once per file. Average the repeats into one point, otherwise the labels land
    on top of each other.
    """
    exact = [r["qps"] for r in rows if r["index"] == "knn"]

    by_index: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        if r["index"] == "knn":
            continue
        key = label_params(r["params"])
        by_index.setdefault(r["index"], {}).setdefault(key, []).append(r)

    series = []
    for i, (name, sweeps) in enumerate(sorted(by_index.items())):
        points = [
            {
                "params": group[0]["params"],
                "qps": sum(r["qps"] for r in group) / len(group),
                "recall": sum(r["recall"] for r in group) / len(group),
            }
            for group in sweeps.values()
        ]
        points.sort(key=lambda r: r["recall"])
        series.append(
            {"name": name, "color": COLORS[i % len(COLORS)], "points": points}
        )

    return series, sum(exact) / len(exact)


def main() -> None:
    series, knn_qps = load_series(load_rows())

    fig, ax = plt.subplots(figsize=(10, 6))

    for s in series:
        pts = s["points"]
        ax.plot(
            [r["recall"] for r in pts],
            [r["qps"] for r in pts],
            marker="o",
            color=s["color"],
            linewidth=1.5,
            label=s["name"],
            zorder=2,
        )
        # Stagger the labels vertically: the best indexes bunch up in the last
        # percent of the recall axis, where a fixed offset would stack them.
        for i, r in enumerate(pts):
            ax.annotate(
                label_params(r["params"]),
                (r["recall"], r["qps"]),
                textcoords="offset points",
                xytext=(-8, 12 + 13 * (i % 3)),
                ha="right",
                fontsize=8,
                color=s["color"],
            )

    ax.scatter(
        [1.0],
        [knn_qps],
        marker="*",
        s=240,
        color=KNN_COLOR,
        zorder=3,
        label="knn (exact)",
    )
    ax.annotate(
        f"knn {knn_qps:.0f} qps",
        (1.0, knn_qps),
        textcoords="offset points",
        xytext=(-10, 10),
        ha="right",
        fontsize=8,
        color=KNN_COLOR,
    )

    ax.set_yscale("log")
    ax.set_xlim(0, 1.0)
    # Headroom so the staggered param labels are not clipped by the frame.
    ax.set_ylim(55, 7000)
    ax.set_xlabel("recall@10 vs exact KNN")
    ax.set_ylabel("Throughput (queries per second, log scale)")
    ax.set_title("Throughput vs recall — GloVe 6B 100d, 20K vectors, 200 queries")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
