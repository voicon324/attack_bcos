#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
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
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_DIR = ROOT / "artifacts" / "analysis" / "camopatch_6models_no_vitc_b_latest"
DEFAULT_OUTPUT_DIR = DEFAULT_SUMMARY_DIR / "charts_position_bars"

MODEL_ORDER = ["resnet18", "resnet50", "densenet121", "convnext_tiny", "convnext_base", "vitc_s", "vitc_b"]
POSITION_ORDER = ["random", "bcos_top1", "gradcam"]
POSITION_LABELS = {
    "random": "Random",
    "bcos_top1": "Top1 B-cos",
    "gradcam": "Grad-CAM",
}
POSITION_COLORS = {
    "random": ("#A3BEFA", "#2E4780"),
    "bcos_top1": ("#FFE15B", "#736422"),
    "gradcam": ("#F390CA", "#8A3A6F"),
}
MODE_LABELS = {
    "fixed": "Fixed / no move",
    "movable": "Movable",
}
DENOMINATOR_FILES = {
    "clean": "summary_clean_correct.csv",
    "all": "summary_all_images.csv",
}
DENOMINATOR_LABELS = {
    "clean": "Clean-correct images only",
    "all": "All images",
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
        description="Save one PNG grouped bar chart per CamoPatch config.",
    )
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--denominators",
        nargs="+",
        choices=sorted(DENOMINATOR_FILES),
        default=["clean", "all"],
        help="Denominator panels to include in each config image.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--format", choices=("png", "svg"), default="png")
    parser.add_argument(
        "--attack",
        default="camopatch",
        help="Attack to plot from aggregated summaries: camopatch, patchrs, or all.",
    )
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
            "patch.linewidth": 1.0,
        },
    )


def read_summary(path: Path, attack: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, Any]] = []
    for row in rows:
        row_attack = (row.get("attack") or "camopatch").strip() or "camopatch"
        if attack != "all" and row_attack != attack:
            continue
        parsed.append(
            {
                "attack": row_attack,
                "position_mode": row["position_mode"],
                "move_allowed": int(float(row["move_allowed"])),
                "model": row["model"],
                "patch_size": int(float(row["patch_size"])),
                "linf": row["linf"],
                "eps_linf": row["eps_linf"],
                "position": row["position"],
                "queries": int(float(row["queries"])),
                "jobs": int(float(row["jobs"])),
                "images": int(float(row["images"])),
                "successes": int(float(row["successes"])),
                "success_rate": float(row["success_rate"]),
            }
        )
    return parsed


def load_datasets(summary_dir: Path, denominators: list[str], attack: str) -> dict[str, list[dict[str, Any]]]:
    return {name: read_summary(summary_dir / DENOMINATOR_FILES[name], attack) for name in denominators}


def config_key(row: dict[str, Any]) -> tuple[str, int, str, int]:
    return (row["position_mode"], int(row["patch_size"]), str(row["linf"]), int(row["queries"]))


def config_sort_key(config: tuple[str, int, str, int]) -> tuple[int, int, int, int]:
    mode, patch_size, linf, queries = config
    linf_value = int(linf.split("/", 1)[0]) if "/" in linf else 999
    return ({"fixed": 0, "movable": 1}.get(mode, 9), patch_size, linf_value, queries)


def config_slug(config: tuple[str, int, str, int]) -> str:
    mode, patch_size, linf, queries = config
    return f"{mode}_s{patch_size}_linf{linf.replace('/', '_')}_q{queries}"


def config_title(config: tuple[str, int, str, int]) -> str:
    mode, patch_size, linf, queries = config
    return f"{MODE_LABELS.get(mode, mode)} | patch {patch_size}x{patch_size} | L_inf {linf} | {queries:,} queries"


def group_rows(rows: list[dict[str, Any]], config: tuple[str, int, str, int]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if config_key(row) == config:
            grouped[(row["model"], row["position"])] = row
    return grouped


def models_for_config(datasets: dict[str, list[dict[str, Any]]], config: tuple[str, int, str, int]) -> list[str]:
    models = {
        row["model"]
        for rows in datasets.values()
        for row in rows
        if config_key(row) == config
    }
    ordered = [model for model in MODEL_ORDER if model in models]
    ordered.extend(sorted(models - set(ordered)))
    return ordered


def draw_panel(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    config: tuple[str, int, str, int],
    models: list[str],
    denominator: str,
) -> None:
    grouped = group_rows(rows, config)
    x = np.arange(len(models), dtype=float)
    width = 0.22
    offsets = {
        "random": -width,
        "bcos_top1": 0.0,
        "gradcam": width,
    }
    for position_index, position in enumerate(POSITION_ORDER):
        xs: list[float] = []
        ys: list[float] = []
        labels: list[str] = []
        for idx, model in enumerate(models):
            row = grouped.get((model, position))
            if row is None:
                continue
            xs.append(x[idx] + offsets[position])
            ys.append(row["success_rate"])
            labels.append(f"{row['success_rate'] * 100:.0f}%")
        if not xs:
            continue
        face, edge = POSITION_COLORS[position]
        bars = ax.bar(
            xs,
            ys,
            width=width * 0.9,
            label=POSITION_LABELS[position],
            color=face,
            edgecolor=edge,
            linewidth=1.0,
        )
        for bar, label in zip(bars, labels):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(bar.get_height() + 0.018 + position_index * 0.026, 1.105),
                label,
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=TOKENS["ink"],
                fontfamily=MONO_FONT_FAMILY[0],
            )

    ax.set_title(DENOMINATOR_LABELS[denominator], fontsize=10.5, color=TOKENS["ink"], pad=8)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("Success rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=22, ha="right")
    ax.tick_params(axis="both", labelsize=8.5, colors=TOKENS["muted"])
    ax.grid(axis="y", color=TOKENS["grid"], linewidth=0.8)
    ax.grid(axis="x", visible=False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(TOKENS["axis"])
    sns.despine(ax=ax)


def write_chart_map(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "config",
        "position_mode",
        "patch_size",
        "linf",
        "queries",
        "image_path",
        "denominators",
        "models",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_config_chart(
    datasets: dict[str, list[dict[str, Any]]],
    config: tuple[str, int, str, int],
    output_dir: Path,
    fmt: str,
    dpi: int,
) -> Path:
    denominators = list(datasets)
    models = models_for_config(datasets, config)
    if not models:
        raise ValueError(f"No rows for config {config}")

    fig_width = 7.2 * len(denominators)
    fig, axes = plt.subplots(
        1,
        len(denominators),
        figsize=(fig_width, 5.2),
        sharey=True,
        squeeze=False,
    )
    fig.patch.set_facecolor(TOKENS["surface"])
    title = config_title(config)
    subtitle = (
        "Grouped bars compare patch init/location rules by model. "
        "Grad-CAM bars are omitted where that job/model is not present."
    )
    fig.suptitle(textwrap.fill(title, width=110), x=0.055, y=0.985, ha="left", fontsize=14, fontweight="semibold", color=TOKENS["ink"])
    fig.text(0.055, 0.925, textwrap.fill(subtitle, width=140), ha="left", va="top", fontsize=9, color=TOKENS["muted"])
    for ax, denominator in zip(axes[0], denominators):
        draw_panel(ax, datasets[denominator], config, models, denominator)

    handles = [
        Patch(facecolor=POSITION_COLORS[position][0], edgecolor=POSITION_COLORS[position][1], label=POSITION_LABELS[position])
        for position in POSITION_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.055, 0.875),
        frameon=False,
        ncol=3,
        fontsize=9,
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.78, bottom=0.16, wspace=0.12)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{config_slug(config)}.{fmt}"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    use_chart_theme()
    summary_dir = args.summary_dir.resolve()
    output_dir = (args.output_dir or summary_dir / "charts_position_bars").resolve()
    datasets = load_datasets(summary_dir, args.denominators, args.attack)
    configs = sorted(
        {
            config_key(row)
            for rows in datasets.values()
            for row in rows
        },
        key=config_sort_key,
    )

    chart_rows: list[dict[str, Any]] = []
    for config in configs:
        output_path = save_config_chart(datasets, config, output_dir, args.format, args.dpi)
        models = models_for_config(datasets, config)
        chart_rows.append(
            {
                "config": config_slug(config),
                "position_mode": config[0],
                "patch_size": config[1],
                "linf": config[2],
                "queries": config[3],
                "image_path": str(output_path),
                "denominators": ";".join(args.denominators),
                "models": ";".join(models),
            }
        )

    write_chart_map(output_dir / "chart_map.csv", chart_rows)
    print(f"output_dir={output_dir}")
    print(f"charts={len(chart_rows)}")
    for row in chart_rows:
        print(row["image_path"])


if __name__ == "__main__":
    main()
