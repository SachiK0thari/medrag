"""Draw the cost-quality chart embedded in the README.

Reads `data/eval_results.csv` (one row per config and question) and plots,
per config, mean retrieval quality against the two things it costs:

    left panel    mean billed USD per query
    right panel   p50 latency, log scale

The panels share one y axis. The shaded band is the hybrid + rerank baseline
plus or minus the run-to-run noise of a single agentic run, so a point inside
it is not distinguishable from the baseline.

Writes a light and a dark PNG; the README picks one by colour scheme:

    docs/eval_cost_quality.png
    docs/eval_cost_quality_dark.png

No network, no models. Label offsets are hand-placed for the committed run,
so re-check the image after a new eval.

Usage:
    python scripts/plot_eval.py
    python scripts/plot_eval.py --metric ndcg_at_10
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator, NullLocator

BASELINE = "hybrid_reranked"

# Recall@10 gap between two agentic configs that are identical up to the first
# merge (0.510 against 0.465): planner sampling noise at temperature 0.1.
# See docs/eval_findings.md, "Reflection is not the cause".
RECALL_NOISE = 0.045

METRIC_LABELS = {
    "recall_at_10": "Recall@10",
    "ndcg_at_10": "nDCG@10",
    "mrr": "MRR",
}

SINGLE_SHOT, ABLATION, SHIPPED = "single_shot", "ablation", "shipped"

# group -> (legend label, marker). Shape repeats what colour says, so the
# groups stay apart in greyscale and for colour-blind readers.
GROUPS = {
    SINGLE_SHOT: ("Single-shot pipeline (no agent)", "o"),
    ABLATION: ("Agentic v1 and its one-knob variants", "s"),
    SHIPPED: ("Agentic v2 (ships)", "D"),
}


@dataclass(frozen=True)
class Point:
    label: str
    group: str
    # Label offset in points from the marker, per panel: (dx, dy).
    cost_offset: tuple[float, float]
    latency_offset: tuple[float, float]


POINTS = {
    "bm25_only": Point("BM25 only", SINGLE_SHOT, (9, -3), (9, -3)),
    "dense_only": Point("Dense only", SINGLE_SHOT, (9, -9), (9, -8)),
    "hybrid_rrf": Point("Hybrid (RRF)", SINGLE_SHOT, (-9, 4), (9, 8)),
    "hybrid_reranked": Point("Hybrid + rerank", SINGLE_SHOT, (9, 10), (-9, -10)),
    "agentic": Point("Agentic v1", ABLATION, (-9, 5), (-9, 5)),
    "agentic_keep_orig": Point("v1 + verbatim question", ABLATION, (-9, -8), (-9, -8)),
    "agentic_no_reflect": Point("v1 without reflection", ABLATION, (9, -3), (-9, 5)),
    "agentic_rerank_orig": Point("v1 + pool re-scored", ABLATION, (-2, 11), (-7, 11)),
    "agentic_faithful": Point("v1 + v2 planner", ABLATION, (-9, 5), (-9, 9)),
    "agentic_v2": Point("Agentic v2", SHIPPED, (4, -13), (9, -9)),
}


@dataclass(frozen=True)
class Theme:
    surface: str
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    axis: str
    band: str
    colors: dict[str, str]


# Colours were checked for colour-blind separation as a set of three, on each
# surface. Keep the two themes in step if either changes.
LIGHT = Theme(
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    ink_muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    band="#f0efec",
    colors={SINGLE_SHOT: "#2a78d6", ABLATION: "#eb6834", SHIPPED: "#1baf7a"},
)
DARK = Theme(
    surface="#1a1a19",
    ink="#f0efec",
    ink_secondary="#c3c2b7",
    ink_muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    band="#2c2c2a",
    colors={SINGLE_SHOT: "#3987e5", ABLATION: "#d95926", SHIPPED: "#199e70"},
)


def summarise(path: Path, metric: str) -> dict[str, dict[str, float]]:
    """config -> mean metric, mean USD per query, p50 latency in seconds."""
    by_config: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            by_config[row["config"]].append(row)
    return {
        cfg: {
            "quality": statistics.fmean(float(r[metric]) for r in rows),
            "cost": statistics.fmean(float(r["cost_usd"]) for r in rows),
            "latency": statistics.median(float(r["latency_ms"]) for r in rows) / 1000,
        }
        for cfg, rows in by_config.items()
    }


def _style_axes(ax: Axes, theme: Theme) -> None:
    ax.set_facecolor(theme.surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.axis)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color=theme.grid, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=theme.ink_muted, labelsize=9, length=0, pad=6)


def _draw_panel(
    ax: Axes,
    stats: dict[str, dict[str, float]],
    x_key: str,
    theme: Theme,
    band: float | None,
) -> None:
    _style_axes(ax, theme)
    base = stats[BASELINE]["quality"]
    if band is not None:
        ax.axhspan(base - band, base + band, color=theme.band, linewidth=0, zorder=1)
    ax.axhline(base, color=theme.axis, linewidth=1, zorder=1.5)

    for cfg, point in POINTS.items():
        if cfg not in stats:
            continue
        x, y = stats[cfg][x_key], stats[cfg]["quality"]
        ax.scatter(
            x,
            y,
            s=62,
            marker=GROUPS[point.group][1],
            color=theme.colors[point.group],
            edgecolors=theme.surface,
            linewidths=1.5,
            zorder=3,
        )
        dx, dy = point.cost_offset if x_key == "cost" else point.latency_offset
        ax.annotate(
            point.label,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx > 0 else "right" if dx < 0 else "center",
            va="center",
            fontsize=8.5,
            color=theme.ink if point.group == SHIPPED else theme.ink_secondary,
            fontweight="bold" if point.group == SHIPPED else "normal",
            zorder=4,
        )


def draw(stats: dict[str, dict[str, float]], metric: str, theme: Theme, out: Path) -> None:
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
    fig, (ax_cost, ax_lat) = plt.subplots(
        1, 2, figsize=(9.6, 5.0), sharey=True, facecolor=theme.surface
    )
    # The noise estimate was measured on recall, so the band is drawn only there.
    band = RECALL_NOISE if metric == "recall_at_10" else None
    _draw_panel(ax_cost, stats, "cost", theme, band)
    _draw_panel(ax_lat, stats, "latency", theme, band)

    ax_cost.set_xlim(0.00018, 0.00082)
    ax_cost.xaxis.set_major_locator(FixedLocator([0.0002, 0.0004, 0.0006, 0.0008]))
    ax_cost.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.4f}"))
    ax_cost.set_xlabel(
        "Cost per query (mean, USD as billed)", color=theme.ink_secondary, fontsize=9.5, labelpad=8
    )

    ax_lat.set_xscale("log")
    ax_lat.set_xlim(0.8, 45)
    ax_lat.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 40]))
    ax_lat.xaxis.set_minor_locator(NullLocator())
    ax_lat.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} s"))
    ax_lat.set_xlabel(
        "Latency (p50, log scale)", color=theme.ink_secondary, fontsize=9.5, labelpad=8
    )

    label = METRIC_LABELS[metric]
    values = [v["quality"] for v in stats.values()]
    top = max(values) if band is None else max(max(values), stats[BASELINE]["quality"] + band)
    ax_cost.set_ylim(math.floor((min(values) - 0.02) * 20) / 20, top + 0.025)
    ax_cost.yaxis.set_major_locator(MultipleLocator(0.05))
    ax_cost.set_ylabel(label, color=theme.ink_secondary, fontsize=9.5, labelpad=8)
    ax_cost.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))

    if band is not None:
        base = stats[BASELINE]["quality"]
        ax_lat.annotate(
            f"baseline ± {band} run-to-run noise",
            (1.0, base + band),
            xycoords=("axes fraction", "data"),
            xytext=(-4, 4),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8,
            color=theme.ink_muted,
        )

    fig.suptitle(
        f"{label} against what each configuration costs",
        x=0.065,
        y=0.975,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=theme.ink,
    )
    fig.text(
        0.065,
        0.905,
        "33 questions, one run per configuration. Higher is better; further left is cheaper or faster.",
        ha="left",
        fontsize=9.5,
        color=theme.ink_secondary,
    )
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=marker,
            markersize=7.5,
            markerfacecolor=theme.colors[group],
            markeredgecolor=theme.surface,
            label=text,
        )
        for group, (text, marker) in GROUPS.items()
    ]
    legend = fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.055, 0.885),
        ncol=3,
        frameon=False,
        fontsize=9,
        handletextpad=0.3,
        columnspacing=1.8,
    )
    for text in legend.get_texts():
        text.set_color(theme.ink_secondary)

    fig.subplots_adjust(left=0.075, right=0.975, top=0.78, bottom=0.13, wspace=0.08)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=theme.surface)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=Path("data/eval_results.csv"))
    parser.add_argument("--metric", choices=sorted(METRIC_LABELS), default="recall_at_10")
    parser.add_argument("--out", type=Path, default=Path("docs/eval_cost_quality.png"))
    args = parser.parse_args()

    stats = summarise(args.results, args.metric)
    missing = sorted(set(stats) - set(POINTS))
    if missing:
        print(f"warning: no label defined for {missing}; they are not plotted")

    dark_out = args.out.with_name(f"{args.out.stem}_dark{args.out.suffix}")
    draw(stats, args.metric, LIGHT, args.out)
    draw(stats, args.metric, DARK, dark_out)
    print(f"wrote {args.out} and {dark_out}")


if __name__ == "__main__":
    main()
