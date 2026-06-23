#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_DIR = ROOT / "artifacts" / "analysis" / "camopatch_6models_no_vitc_b_latest"

MODEL_ORDER = ["resnet18", "resnet50", "densenet121", "convnext_tiny", "convnext_base", "vitc_s", "vitc_b"]
POSITION_ORDER_BY_MODE = {
    "fixed": ["random", "bcos_top1", "gradcam"],
    "movable": ["random", "bcos_top1"],
}
POSITION_LABELS = {
    "random": "Random",
    "bcos_top1": "Top1 B-cos",
    "gradcam": "Grad-CAM",
}
POSITION_STYLE = {
    "random": {"color": "#5477C4", "linestyle": "-", "marker": "o"},
    "bcos_top1": {"color": "#B8A037", "linestyle": "--", "marker": "s"},
    "gradcam": {"color": "#BD569B", "linestyle": ":", "marker": "^"},
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
            "Save one CamoPatch query-curve image per config/model. "
            "Each image compares available patch position rules over query count."
        ),
    )
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--position-mode", choices=sorted(POSITION_ORDER_BY_MODE), default="movable")
    parser.add_argument(
        "--denominators",
        nargs="+",
        choices=sorted(DENOMINATOR_FILES),
        default=["clean", "all"],
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--format", choices=("png", "svg"), default="png")
    parser.add_argument(
        "--query-scale",
        choices=("linear", "symlog"),
        default="symlog",
        help="Scale for the query axis. symlog preserves query 0 while using log-like spacing after --symlog-linthresh.",
    )
    parser.add_argument(
        "--symlog-linthresh",
        type=float,
        default=10.0,
        help="Linear range around zero for --query-scale symlog.",
    )
    parser.add_argument(
        "--curve-style",
        choices=("smooth", "step"),
        default="smooth",
        help="Use smooth monotone interpolation for presentation, or exact stair-step curves.",
    )
    parser.add_argument(
        "--smooth-points",
        type=int,
        default=900,
        help="Number of dense points used for smooth curves.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=31,
        help="Odd rolling window over dense log-query points for presentation smoothing. Use 1 to disable.",
    )
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
        },
    )


def read_query_curve(path: Path, position_mode: str, attack: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, Any]] = []
    for row in rows:
        row_attack = (row.get("attack") or "camopatch").strip() or "camopatch"
        if attack != "all" and row_attack != attack:
            continue
        if row["position_mode"] != position_mode:
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
                "first_success_query": int(float(row["first_success_query"])),
                "new_successes": int(float(row["new_successes"])),
                "cumulative_successes": int(float(row["cumulative_successes"])),
                "denominator_images": int(float(row["denominator_images"])),
                "success_rate": float(row["success_rate"]),
            }
        )
    return parsed


def load_datasets(summary_dir: Path, denominators: list[str], position_mode: str, attack: str) -> dict[str, list[dict[str, Any]]]:
    return {
        denominator: read_query_curve(summary_dir / DENOMINATOR_FILES[denominator], position_mode, attack)
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


def available_positions_for_key(
    datasets: dict[str, list[dict[str, Any]]],
    key: tuple[str, int, str, int, str],
    positions: list[str],
) -> list[str]:
    available: list[str] = []
    for position in positions:
        if any(rows_for(rows, key, position) for rows in datasets.values()):
            available.append(position)
    return available


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


def _pchip_endpoint_slope(h0: float, h1: float, m0: float, m1: float) -> float:
    slope = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if np.sign(slope) != np.sign(m0):
        return 0.0
    if np.sign(m0) != np.sign(m1) and abs(slope) > abs(3 * m0):
        return 3 * m0
    return float(slope)


def _pchip_slopes(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    h = np.diff(x_values)
    delta = np.diff(y_values) / h
    slopes = np.zeros_like(y_values)
    if len(y_values) == 2:
        slopes[:] = delta[0]
        return slopes

    slopes[0] = _pchip_endpoint_slope(h[0], h[1], delta[0], delta[1])
    slopes[-1] = _pchip_endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])
    for idx in range(1, len(y_values) - 1):
        left = delta[idx - 1]
        right = delta[idx]
        if left == 0 or right == 0 or np.sign(left) != np.sign(right):
            slopes[idx] = 0.0
            continue
        w1 = 2 * h[idx] + h[idx - 1]
        w2 = h[idx] + 2 * h[idx - 1]
        slopes[idx] = (w1 + w2) / (w1 / left + w2 / right)
    return slopes


def smooth_monotone_values(y_values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y_values) < 5:
        return y_values
    if window % 2 == 0:
        window += 1
    window = min(window, len(y_values) if len(y_values) % 2 == 1 else len(y_values) - 1)
    if window < 3:
        return y_values

    half = window // 2
    ramp = np.arange(1, half + 2, dtype=float)
    kernel = np.concatenate([ramp, ramp[-2::-1]])
    kernel /= kernel.sum()
    padded = np.pad(y_values, (half, half), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def smooth_curve_points(
    x_values: list[int],
    y_values: list[float],
    max_queries: int,
    points: int,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(x_values) < 3:
        return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)

    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=float)
    x_array, unique_indices = np.unique(x_array, return_index=True)
    y_array = y_array[unique_indices]
    if len(x_array) < 3:
        return x_array, y_array

    transformed_x = np.log1p(x_array)
    dense_count = max(points, len(x_array) * 3, 2)
    dense_x_t = np.linspace(transformed_x[0], transformed_x[-1], dense_count)
    slopes = _pchip_slopes(transformed_x, y_array)
    interval_idx = np.searchsorted(transformed_x, dense_x_t, side="right") - 1
    interval_idx = np.clip(interval_idx, 0, len(transformed_x) - 2)

    x0 = transformed_x[interval_idx]
    x1 = transformed_x[interval_idx + 1]
    y0 = y_array[interval_idx]
    y1 = y_array[interval_idx + 1]
    h = x1 - x0
    t = (dense_x_t - x0) / h
    t2 = t * t
    t3 = t2 * t
    dense_y = (
        (2 * t3 - 3 * t2 + 1) * y0
        + (t3 - 2 * t2 + t) * h * slopes[interval_idx]
        + (-2 * t3 + 3 * t2) * y1
        + (t3 - t2) * h * slopes[interval_idx + 1]
    )
    dense_y = smooth_monotone_values(np.clip(dense_y, 0.0, 1.0), window)
    dense_y = np.maximum.accumulate(np.minimum(dense_y, y_array[-1]))
    dense_y[0] = y_array[0]
    dense_y[-1] = y_array[-1]

    dense_x = np.expm1(dense_x_t)
    dense_x[0] = 0.0
    dense_x[-1] = float(max_queries)
    return dense_x, dense_y


def draw_panel(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    key: tuple[str, int, str, int, str],
    denominator: str,
    positions: list[str],
    query_scale: str,
    symlog_linthresh: float,
    curve_style: str,
    smooth_points: int,
    smooth_window: int,
) -> None:
    max_queries = key[3]
    max_rate = 0.0
    for position in positions:
        series_rows = rows_for(rows, key, position)
        if not series_rows:
            continue
        x_values, y_values, denom_images, successes = curve_points(series_rows, max_queries)
        max_rate = max(max_rate, max(y_values) if y_values else 0.0)
        style = POSITION_STYLE[position]
        label = POSITION_LABELS[position]
        if denom_images:
            label = f"{label} ({successes}/{denom_images}, {y_values[-1] * 100:.1f}%)"
        if curve_style == "smooth":
            plot_x, plot_y = smooth_curve_points(x_values, y_values, max_queries, smooth_points, smooth_window)
            ax.plot(
                plot_x,
                plot_y,
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.9,
                solid_capstyle="round",
                solid_joinstyle="round",
                dash_capstyle="round",
                dash_joinstyle="round",
            )
            marker_source_x = np.asarray([], dtype=float)
            marker_source_y = np.asarray([], dtype=float)
        else:
            ax.step(
                x_values,
                y_values,
                where="post",
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.7,
            )
            marker_source_x = np.asarray(x_values, dtype=float)
            marker_source_y = np.asarray(y_values, dtype=float)
        if len(marker_source_x) > 2:
            marker_idx = np.linspace(1, len(marker_source_x) - 1, num=min(9, len(marker_source_x) - 1), dtype=int)
            ax.plot(
                marker_source_x[marker_idx],
                marker_source_y[marker_idx],
                linestyle="none",
                marker=style["marker"],
                markersize=3.1,
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
            )
    upper = min(1.0, max(0.1, max_rate * 1.12))
    ax.set_xlim(0, max_queries)
    ax.set_ylim(0, upper)
    ax.set_title(DENOMINATOR_LABELS[denominator], fontsize=10.5, color=TOKENS["ink"], pad=8)
    if query_scale == "symlog":
        ax.set_xscale("symlog", linthresh=symlog_linthresh, linscale=1.0)
        ticks = [0, 1, 10, 100, 1000, max_queries]
        ticks = sorted({tick for tick in ticks if 0 <= tick <= max_queries})
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _pos: "0" if value == 0 else f"{int(value):,}")
        )
        ax.set_xlabel(f"Query (symlog, linear <= {symlog_linthresh:g})")
    else:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
        ax.set_xlabel("Query")
    ax.set_ylabel("Cumulative success rate")
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
    positions: list[str],
    output_dir: Path,
    fmt: str,
    dpi: int,
    query_scale: str,
    symlog_linthresh: float,
    curve_style: str,
    smooth_points: int,
    smooth_window: int,
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
    if curve_style == "smooth":
        curve_note = "Presentation-smoothed monotone curves show cumulative attack success by first-success query; final rates use exact counts."
    else:
        curve_note = "Step curves show exact cumulative attack success by first-success query."
    position_labels = ", ".join(POSITION_LABELS[position] for position in positions)
    subtitle = (
        f"{curve_note} Each panel compares available {MODE_LABELS.get(key[0], key[0]).lower()} position rules "
        f"({position_labels}); "
        f"query axis uses {query_scale} scale."
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
        draw_panel(
            ax,
            datasets[denominator],
            key,
            denominator,
            positions,
            query_scale,
            symlog_linthresh,
            curve_style,
            smooth_points,
            smooth_window,
        )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.79, bottom=0.14, wspace=0.14)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{chart_slug(key)}.{fmt}"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_chart_map(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "chart",
        "position_mode",
        "patch_size",
        "linf",
        "queries",
        "model",
        "image_path",
        "denominators",
        "query_scale",
        "symlog_linthresh",
        "curve_style",
        "smooth_points",
        "smooth_window",
        "positions",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    use_chart_theme()
    summary_dir = args.summary_dir.resolve()
    output_dir = (args.output_dir or summary_dir / f"charts_{args.position_mode}_query_curves").resolve()
    datasets = load_datasets(summary_dir, args.denominators, args.position_mode, args.attack)
    positions = POSITION_ORDER_BY_MODE[args.position_mode]
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
        chart_positions = available_positions_for_key(datasets, key, positions)
        output_path = save_chart(
            datasets,
            key,
            chart_positions,
            output_dir,
            args.format,
            args.dpi,
            args.query_scale,
            args.symlog_linthresh,
            args.curve_style,
            args.smooth_points,
            args.smooth_window,
        )
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
                "query_scale": args.query_scale,
                "symlog_linthresh": args.symlog_linthresh,
                "curve_style": args.curve_style,
                "smooth_points": args.smooth_points,
                "smooth_window": args.smooth_window,
                "positions": ";".join(chart_positions),
            }
        )
    write_chart_map(output_dir / "chart_map.csv", chart_rows)
    print(f"output_dir={output_dir}")
    print(f"charts={len(chart_rows)}")
    for row in chart_rows:
        print(row["image_path"])


if __name__ == "__main__":
    main()
