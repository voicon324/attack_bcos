#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_DIR = ROOT / "artifacts" / "analysis" / "camopatch_6models_no_vitc_b_latest"

MODEL_ORDER = ["resnet18", "resnet50", "densenet121", "convnext_tiny", "convnext_base", "vitc_s", "vitc_b"]
POSITION_ORDER = ["random", "bcos_top1"]
POSITION_LABELS = {
    "random": "Random init",
    "bcos_top1": "Top1 B-cos init",
}
POSITION_STYLE = {
    "random": {"color": "#5477C4", "linestyle": "-", "marker": "o"},
    "bcos_top1": {"color": "#B8A037", "linestyle": "--", "marker": "s"},
}
DENOMINATOR_FILES = {
    "clean": "success_by_query_clean_correct.csv",
    "all": "success_by_query_all_images.csv",
}
DENOMINATOR_LABELS = {
    "clean": "Clean-correct images only",
    "all": "All images",
}
MODE_LABELS = {
    "fixed": "Fixed / no move",
    "movable": "Movable",
}

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

FONT_FAMILY = ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
MONO_FONT_FAMILY = ["DejaVu Sans Mono", "Menlo", "Consolas", "monospace"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save one CamoPatch movable query-curve image per config/model. "
            "Each image compares random vs top1 init over query count."
        ),
    )
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--position-mode", default="movable")
    parser.add_argument(
        "--denominators",
        nargs="+",
        choices=sorted(DENOMINATOR_FILES),
        default=["clean", "all"],
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--format", choices=("png", "svg"), default="png")
    return parser.parse_args()


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "figure.edgecolor": "none",
            "savefig.facecolor": TOKENS["surface"],
            "savefig.edgecolor": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.monospace": MONO_FONT_FAMILY,
        },
    )


def read_query_curve(path: Path, position_mode: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, Any]] = []
    for row in rows:
        if row["position_mode"] != position_mode:
            continue
        parsed.append(
            {
                "position_mode": row["position_mode"],
                "move_allowed": int(float(row["move_allowed"])),
                "model": row["model"],
                "patch_size": int(float(row["patch_size"])),
                "linf": row["linf"],
                "eps_linf": row["eps_linf"],
                "position": row["position"],
                "queries": int(float(row["queries"])),
                "first_success_query": int(float(row["first_success_query"])),
                "new_successes": int(float(row["new_successes"])),
                "cumulative_successes": int(float(row["cumulative_successes"])),
                "denominator_images": int(float(row["denominator_images"])),
                "success_rate": float(row["success_rate"]),
            }
        )
    return parsed


def load_datasets(summary_dir: Path, denominators: list[str], position_mode: str) -> dict[str, list[dict[str, Any]]]:
    return {
        denominator: read_query_curve(summary_dir / DENOMINATOR_FILES[denominator], position_mode)
        for denominator in denominators
    }


def config_key(row: dict[str, Any]) -> tuple[str, int, str, int]:
    return (row["position_mode"], int(row["patch_size"]), str(row["linf"]), int(row["queries"]))


def full_key(row: dict[str, Any]) -> tuple[str, int, str, int, str]:
    return (*config_key(row), row["model"])


def config_sort_key(config: tuple[str, int, str, int, str]) -> tuple[int, int, int, int, int]:
    mode, patch_size, linf, queries, model = config
    linf_value = int(linf.split("/", 1)[0]) if "/" in linf else 999
    return (
        {"fixed": 0, "movable": 1}.get(mode, 9),
        patch_size,
        linf_value,
        queries,
        MODEL_ORDER.index(model) if model in MODEL_ORDER else 999,
    )


def chart_slug(key: tuple[str, int, str, int, str]) -> str:
    mode, patch_size, linf, queries, model = key
    return f"{mode}_s{patch_size}_linf{linf.replace('/', '_')}_q{queries}_{model}"


def chart_title(key: tuple[str, int, str, int, str]) -> str:
    mode, patch_size, linf, queries, model = key
    return f"{MODE_LABELS.get(mode, mode)} | {model} | patch {patch_size}x{patch_size} | L_inf {linf} | {queries:,} queries"


def rows_for(
    rows: list[dict[str, Any]],
    key: tuple[str, int, str, int, str],
    position: str,
) -> list[dict[str, Any]]:
    mode, patch_size, linf, queries, model = key
    selected = [
        row
        for row in rows
        if row["position_mode"] == mode
        and row["patch_size"] == patch_size
        and row["linf"] == linf
        and row["queries"] == queries
        and row["model"] == model
        and row["position"] == position
    ]
    return sorted(selected, key=lambda item: item["first_success_query"])


def curve_points(rows: list[dict[str, Any]], max_queries: int) -> tuple[list[int], list[float], int, int]:
    if not rows:
        return [0, max_queries], [0.0, 0.0], 0, 0
    x_values = [0]
    y_values = [0.0]
    for row in rows:
        query = row["first_success_query"]
        rate = row["success_rate"]
        if x_values[-1] == query:
            y_values[-1] = rate
        else:
            x_values.append(query)
            y_values.append(rate)
    if x_values[-1] < max_queries:
        x_values.append(max_queries)
        y_values.append(y_values[-1])
    denominator = rows[-1]["denominator_images"]
    successes = rows[-1]["cumulative_successes"]
    return x_values, y_values, denominator, successes


def draw_panel(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    key: tuple[str, int, str, int, str],
    denominator: str,
) -> None:
    max_queries = key[3]
    max_rate = 0.0
    for position in POSITION_ORDER:
        series_rows = rows_for(rows, key, position)
        x_values, y_values, denom_images, successes = curve_points(series_rows, max_queries)
        max_rate = max(max_rate, max(y_values) if y_values else 0.0)
        style = POSITION_STYLE[position]
        label = POSITION_LABELS[position]
        if denom_images:
            label = f"{label} ({successes}/{denom_images}, {y_values[-1] * 100:.1f}%)"
        ax.step(
            x_values,
            y_values,
            where="post",
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.7,
        )
        if len(x_values) > 2:
            marker_idx = np.linspace(1, len(x_values) - 1, num=min(9, len(x_values) - 1), dtype=int)
            ax.plot(
                [x_values[idx] for idx in marker_idx],
                [y_values[idx] for idx in marker_idx],
                linestyle="none",
                marker=style["marker"],
                markersize=3.2,
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
            )
    upper = min(1.0, max(0.1, max_rate * 1.12))
    ax.set_xlim(0, max_queries)
    ax.set_ylim(0, upper)
    ax.set_title(DENOMINATOR_LABELS[denominator], fontsize=10.5, color=TOKENS["ink"], pad=8)
    ax.set_xlabel("Query")
    ax.set_ylabel("Cumulative success rate")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.tick_params(axis="both", labelsize=8.5, colors=TOKENS["muted"])
    ax.grid(axis="y", color=TOKENS["grid"], linewidth=0.8)
    ax.grid(axis="x", color=TOKENS["grid"], linewidth=0.6, alpha=0.5)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(TOKENS["axis"])
    ax.legend(loc="lower right", frameon=False, fontsize=8.2)
    sns.despine(ax=ax)


def save_chart(
    datasets: dict[str, list[dict[str, Any]]],
    key: tuple[str, int, str, int, str],
    output_dir: Path,
    fmt: str,
    dpi: int,
) -> Path:
    denominators = list(datasets)
    fig, axes = plt.subplots(
        1,
        len(denominators),
        figsize=(7.2 * len(denominators), 5.2),
        sharey=False,
        squeeze=False,
    )
    fig.patch.set_facecolor(TOKENS["surface"])
    title = chart_title(key)
    subtitle = (
        "Step curves show cumulative attack success by first-success query. "
        "Each panel compares movable random init against movable Top1 B-cos init."
    )
    fig.suptitle(
        textwrap.fill(title, width=120),
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=14,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(0.055, 0.925, textwrap.fill(subtitle, width=145), ha="left", va="top", fontsize=9, color=TOKENS["muted"])
    for ax, denominator in zip(axes[0], denominators):
        draw_panel(ax, datasets[denominator], key, denominator)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.79, bottom=0.14, wspace=0.14)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{chart_slug(key)}.{fmt}"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_chart_map(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["chart", "position_mode", "patch_size", "linf", "queries", "model", "image_path", "denominators"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    use_chart_theme()
    summary_dir = args.summary_dir.resolve()
    output_dir = (args.output_dir or summary_dir / "charts_movable_query_curves").resolve()
    datasets = load_datasets(summary_dir, args.denominators, args.position_mode)
    keys = sorted(
        {
            full_key(row)
            for rows in datasets.values()
            for row in rows
        },
        key=config_sort_key,
    )
    chart_rows: list[dict[str, Any]] = []
    for key in keys:
        output_path = save_chart(datasets, key, output_dir, args.format, args.dpi)
        chart_rows.append(
            {
                "chart": chart_slug(key),
                "position_mode": key[0],
                "patch_size": key[1],
                "linf": key[2],
                "queries": key[3],
                "model": key[4],
                "image_path": str(output_path),
                "denominators": ";".join(args.denominators),
            }
        )
    write_chart_map(output_dir / "chart_map.csv", chart_rows)
    print(f"output_dir={output_dir}")
    print(f"charts={len(chart_rows)}")
    for row in chart_rows:
        print(row["image_path"])


if __name__ == "__main__":
    main()
