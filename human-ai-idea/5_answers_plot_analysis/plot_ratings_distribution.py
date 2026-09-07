from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


COLOR_HUMAN  = "#e5b635"
COLOR_GPT4   = "#1e88e5"
COLOR_GPT52  = "#43a047"

LINE_HUMAN   = "#be7b0e"
LINE_GPT4    = "#0d47a1"
LINE_GPT52   = "#2e6f32"


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor":   "white",
            "savefig.facecolor":"white",
            "axes.titlesize":   16,
            "axes.labelsize":   13,
            "xtick.labelsize":  11,
            "ytick.labelsize":  11,
            "legend.fontsize":  11,
            "font.size":        11,
        }
    )


def style_axes(ax, spine_lw: float = 1.0) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.55)
    ax.spines["bottom"].set_alpha(0.55)
    ax.spines["left"].set_linewidth(spine_lw)
    ax.spines["bottom"].set_linewidth(spine_lw)
    ax.set_axisbelow(True)


def load_ratings_by_source(path: Path) -> dict[str, List[Optional[float]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    by_source: dict[str, List[Optional[float]]] = {"human": [], "gpt4": [], "gpt5.2": []}
    for rec in data:
        source = rec.get("source")
        if source in by_source:
            by_source[source].append(rec.get("interest_rating"))
    return by_source


# -----------------------------
# Statistical helpers (same formulas as analyse.py)
# -----------------------------

def proportion_error_pct(k: int, n: int, error_style: str = "sem") -> float:
    if n <= 1:
        return 0.0
    p = k / n
    if error_style == "sd":
        return 100.0 * math.sqrt((n / (n - 1.0)) * p * (1.0 - p))
    if error_style == "sem":
        return 100.0 * math.sqrt(p * (1.0 - p) / (n - 1.0))
    raise ValueError(f"Unknown error_style={error_style!r}; expected 'sd' or 'sem'")


def rating_counts(values: List[Optional[float]]) -> Tuple[np.ndarray, int]:
    cleaned = [int(round(v)) for v in values if v is not None and 1 <= float(v) <= 5]
    counts = np.zeros(5, dtype=int)
    for v in cleaned:
        counts[v - 1] += 1
    return counts, len(cleaned)


def percentages_with_error(
    values: List[Optional[float]],
    error_style: str = "sem",
) -> Tuple[np.ndarray, np.ndarray, int]:
    counts, n = rating_counts(values)
    if n == 0:
        return np.zeros(5, dtype=float), np.zeros((2, 5), dtype=float), 0
    pct = 100.0 * counts.astype(float) / float(n)
    err = np.array(
        [proportion_error_pct(int(k), n, error_style=error_style) for k in counts],
        dtype=float,
    )
    yerr = np.vstack([err, err])
    return pct, yerr, n


# -----------------------------
# Plots
# -----------------------------

def plot_distribution_bar_chart(
    human_vals: List[Optional[float]],
    gpt4_vals: List[Optional[float]],
    gpt52_vals: List[Optional[float]],
    out_plot: Path,
    out_pdf: Path,
    show_plot: bool,
    error_style: str,
) -> None:
    human_pct,  human_yerr,  n_human  = percentages_with_error(human_vals,  error_style=error_style)
    gpt4_pct,   gpt4_yerr,   n_gpt4   = percentages_with_error(gpt4_vals,   error_style=error_style)
    gpt52_pct,  gpt52_yerr,  n_gpt52  = percentages_with_error(gpt52_vals,  error_style=error_style)

    x     = np.arange(1, 6)
    width = 0.24

    TITLE_FS   = 18
    LABEL_FS   = 17
    TICK_FS    = 17
    ANNOT_FS   = 15
    LEGEND_FS  = 17
    SPINE_LW   = 1.8
    ERROR_LW   = 1.8
    CAP_SIZE   = 6
    BAR_EDGE_W = 1.3

    error_kw_human = dict(ecolor=LINE_HUMAN, elinewidth=ERROR_LW,
                          capthick=ERROR_LW, capsize=CAP_SIZE, alpha=0.95)
    error_kw_gpt4  = dict(ecolor=LINE_GPT4,  elinewidth=ERROR_LW,
                          capthick=ERROR_LW, capsize=CAP_SIZE, alpha=0.95)
    error_kw_gpt52 = dict(ecolor=LINE_GPT52, elinewidth=ERROR_LW,
                          capthick=ERROR_LW, capsize=CAP_SIZE, alpha=0.95)

    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    ax.set_axisbelow(True)

    bars_h = ax.bar(
        x - width, human_pct, width=width,
        color=COLOR_HUMAN, alpha=0.94,
        label=f"Human (n={n_human})",
        edgecolor="white", linewidth=BAR_EDGE_W,
        yerr=human_yerr, error_kw=error_kw_human,
        zorder=3,
    )
    bars_4 = ax.bar(
        x, gpt4_pct, width=width,
        color=COLOR_GPT4, alpha=0.94,
        label=f"GPT-4 (n={n_gpt4})",
        edgecolor="white", linewidth=BAR_EDGE_W,
        yerr=gpt4_yerr, error_kw=error_kw_gpt4,
        zorder=3,
    )
    bars_52 = ax.bar(
        x + width, gpt52_pct, width=width,
        color=COLOR_GPT52, alpha=0.94,
        label=f"GPT-5.2 (n={n_gpt52})",
        edgecolor="white", linewidth=BAR_EDGE_W,
        yerr=gpt52_yerr, error_kw=error_kw_gpt52,
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(["1", "2", "3", "4", "5"], fontsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.set_xlabel("Interest value", fontsize=LABEL_FS)
    ax.set_ylabel("Percentage (%)", fontsize=LABEL_FS)
    ax.set_title("Distribution of interest ratings", fontsize=TITLE_FS, pad=14)

    ax.grid(axis="y", linestyle="--", alpha=0.22)
    ax.grid(axis="x", visible=False)
    style_axes(ax, spine_lw=SPINE_LW)

    ax.legend(
        frameon=True, fontsize=LEGEND_FS, ncol=1,
        loc="upper right", framealpha=0.9, edgecolor="0.75",
    )

    ymax = max(
        float(np.max(human_pct  + human_yerr[1])),
        float(np.max(gpt4_pct   + gpt4_yerr[1])),
        float(np.max(gpt52_pct  + gpt52_yerr[1])),
    )
    ax.set_ylim(0, max(20.0, ymax * 1.16 + 0.8))

    for bars in (bars_h, bars_4, bars_52):
        for rect in bars:
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                h / 2,
                f"{h:.1f}%",
                ha="center", va="center",
                fontsize=ANNOT_FS, color="black",
                rotation=90, zorder=4,
            )

    plt.tight_layout()
    plt.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"Saved bar chart to: {out_plot}")
    print(f"Saved bar chart to: {out_pdf}")
    if show_plot:
        plt.show()
    plt.close()


BIN_COLORS = [
    (0.7, 0.7, 0.8, 1.0),
    (0.6, 0.4, 0.8, 0.5),
    (0.9, 0.3, 0.5, 0.3),
    (0.9, 0.5, 0.3, 0.5),
    (0.9, 0.65, 0.0, 0.5),
]


def plot_three_panel_distribution(
    human_vals: List[Optional[float]],
    gpt4_vals: List[Optional[float]],
    gpt52_vals: List[Optional[float]],
    out_plot: Path,
    show_plot: bool,
) -> None:
    """One subplot per source (Human-Written / GPT-4 / GPT-5.2), each showing
    that source's own rating-count distribution as percentages, with the raw
    count labeled inside each bar."""
    panels = [
        ("Human-Written Ideas", human_vals),
        ("GPT-4 Ideas", gpt4_vals),
        ("GPT-5.2 Ideas", gpt52_vals),
    ]
    x = np.arange(1, 6)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)

    for ax, (name, vals) in zip(axes, panels):
        counts, n = rating_counts(vals)
        pct = 100.0 * counts.astype(float) / float(n) if n else np.zeros(5, dtype=float)

        bars = ax.bar(x, pct, color=BIN_COLORS, width=0.6)

        for rect, k in zip(bars, counts):
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2.0, h / 2,
                str(int(k)), ha="center", va="center", fontsize=13, color="black", zorder=4,
            )

        ax.set_title(name, fontsize=13, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["1", "2", "3", "4", "5"], fontsize=14)
        ax.set_xlabel("Interest Level", fontsize=14)
        ax.set_ylim(0, 50)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
        ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_alpha(0.8)
        ax.spines["bottom"].set_alpha(0.8)
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)

    axes[0].set_ylabel("Percentage (%)", fontsize=14)

    plt.tight_layout()
    plt.savefig(out_plot, dpi=300, bbox_inches="tight")
    print(f"Saved three-panel distribution chart to: {out_plot}")
    if show_plot:
        plt.show()
    plt.close()


def main() -> int:
    apply_plot_style()

    parser = argparse.ArgumentParser(
        description="Plot rating-distribution figures directly from all_ratings_results.json."
    )
    parser.add_argument("--input-file", type=str, default="all_ratings_results.json")
    parser.add_argument("--output-bar-plot",   type=str, default="rankings_distribution_all_three.png")
    parser.add_argument("--output-bar-pdf",    type=str, default="Fig6_rankings_distribution_all_three.pdf")
    parser.add_argument("--output-panel-plot", type=str, default="distribution_interest_ratings_human_AI.png")
    parser.add_argument("--error-style", choices=["sem", "sd"], default="sem")
    parser.add_argument("--no-show", action="store_true")
    args, _ = parser.parse_known_args()

    input_path = Path(args.input_file).expanduser().resolve()
    by_source = load_ratings_by_source(input_path)
    human_vals, gpt4_vals, gpt52_vals = by_source["human"], by_source["gpt4"], by_source["gpt5.2"]

    print(f"Loaded {input_path}")
    print(f"  human: {len(human_vals)}, gpt4: {len(gpt4_vals)}, gpt5.2: {len(gpt52_vals)}")

    show_plots = not args.no_show

    plot_distribution_bar_chart(
        human_vals, gpt4_vals, gpt52_vals,
        out_plot=Path(args.output_bar_plot).expanduser().resolve(),
        out_pdf=Path(args.output_bar_pdf).expanduser().resolve(),
        show_plot=show_plots, error_style=args.error_style,
    )
    plot_three_panel_distribution(
        human_vals, gpt4_vals, gpt52_vals,
        out_plot=Path(args.output_panel_plot).expanduser().resolve(),
        show_plot=show_plots,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
