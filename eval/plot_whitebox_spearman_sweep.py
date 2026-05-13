#!/usr/bin/env python3
"""
Create charts from whitebox Spearman sweep results.

The script prefers:
  result_root/merged/<model>/eps_*/results.csv
for each model/epsilon pair that has merged output. Missing model/epsilon pairs
fall back to:
  result_root/chunks/chunk_*/<model>/eps_*/results.csv

It writes a per-model/epsilon summary CSV plus PNG charts.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_RESULT_ROOT = Path("artifacts") / "outputs" / "result" / "whitebox_spearman_model_epsilon_sweep_parallel"
MODEL_ORDER = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
]
METRICS = [
    "spearman",
    "whitebox_before_after_spearman",
    "builtin_before_after_spearman",
    "method_spearman",
    "attack_explain_cosine_distance",
    "attack_explain_mse",
    "mean_abs_diff",
    "max_abs_diff",
    "weights_max_abs_diff",
]


@dataclass
class GroupStats:
    total_rows: int = 0
    ok_rows: int = 0
    error_rows: int = 0
    class_preserved_rows: int = 0
    values: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create summary CSV and charts from whitebox Spearman sweep results.",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help="Sweep output root containing merged/ or chunks/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save charts. Defaults to <result-root>/charts.",
    )
    parser.add_argument(
        "--component",
        default=None,
        help="Component to plot. Defaults to the first component found.",
    )
    parser.add_argument(
        "--alpha-percentile",
        type=float,
        default=None,
        help="Alpha percentile to plot. Defaults to the first value found.",
    )
    parser.add_argument(
        "--primary-metric",
        default="whitebox_before_after_spearman",
        choices=METRICS,
        help="Metric used for the main line plot and heatmap.",
    )
    parser.add_argument(
        "--title-prefix",
        default="Whitebox Spearman Sweep",
        help="Prefix used in chart titles.",
    )
    return parser.parse_args()


def safe_float(value: object) -> Optional[float]:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def model_sort_key(model: str) -> Tuple[int, str]:
    try:
        return MODEL_ORDER.index(model), model
    except ValueError:
        return len(MODEL_ORDER), model


def epsilon_label(epsilon: float) -> str:
    fraction = Fraction(epsilon).limit_denominator(4096)
    if abs(float(fraction) - epsilon) < 1e-10 and fraction.denominator <= 4096:
        return f"{fraction.numerator}/{fraction.denominator}"
    return f"{epsilon:g}"


def discover_result_files(result_root: Path) -> Tuple[List[Path], str]:
    merged_files = sorted((result_root / "merged").glob("*/*/results.csv"))
    chunk_files = sorted((result_root / "chunks").glob("chunk_*/*/eps_*/results.csv"))

    if merged_files and chunk_files:
        merged_keys = {path_model_and_eps(path) for path in merged_files}
        chunk_fallback_files = [
            path
            for path in chunk_files
            if path_model_and_eps(path) not in merged_keys
        ]
        selected_files = sorted([*merged_files, *chunk_fallback_files])
        return (
            selected_files,
            (
                f"hybrid: {len(merged_files)} merged CSV files, "
                f"{len(chunk_fallback_files)} chunk fallback CSV files"
            ),
        )

    if merged_files:
        return merged_files, f"merged ({len(merged_files)} CSV files)"

    if chunk_files:
        return chunk_files, f"chunks ({len(chunk_files)} CSV files)"

    direct_files = sorted(result_root.glob("*/*/results.csv"))
    if direct_files:
        return direct_files, f"direct ({len(direct_files)} CSV files)"

    return [], "none"


def path_model_and_eps(path: Path) -> Tuple[str, str]:
    eps_label = path.parent.name
    model = path.parent.parent.name
    return model, eps_label


def read_rows(paths: Sequence[Path]) -> List[dict]:
    rows: List[dict] = []
    for path in paths:
        path_model, path_eps = path_model_and_eps(path)
        with path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["_source_csv"] = str(path)
                row["_path_model"] = path_model
                row["_path_eps_label"] = path_eps
                rows.append(row)
    return rows


def choose_filter_value(
    rows: Sequence[dict],
    key: str,
    requested: Optional[str],
) -> str:
    values = sorted({str(row.get(key, "")).strip() for row in rows if str(row.get(key, "")).strip()})
    if requested is not None:
        if requested not in values:
            raise ValueError(f"Requested {key}={requested!r}, but available values are: {values}")
        return requested
    if not values:
        raise ValueError(f"No {key} values found in result rows.")
    if len(values) > 1:
        print(f"Found multiple {key} values {values}; using {values[0]!r}.")
    return values[0]


def choose_alpha(rows: Sequence[dict], requested: Optional[float]) -> float:
    values = sorted(
        {
            value
            for row in rows
            if (value := safe_float(row.get("alpha_percentile"))) is not None
        },
    )
    if requested is not None:
        for value in values:
            if abs(value - requested) < 1e-9:
                return value
        raise ValueError(f"Requested alpha_percentile={requested:g}, but available values are: {values}")
    if not values:
        raise ValueError("No alpha_percentile values found in result rows.")
    if len(values) > 1:
        print(f"Found multiple alpha_percentile values {values}; using {values[0]:g}.")
    return values[0]


def aggregate(rows: Sequence[dict]) -> Dict[Tuple[str, float], GroupStats]:
    grouped: Dict[Tuple[str, float], GroupStats] = defaultdict(GroupStats)
    for row in rows:
        model = str(row.get("model") or row.get("_path_model") or "").strip()
        epsilon = safe_float(row.get("attack_epsilon"))
        if not model or epsilon is None:
            continue

        stats = grouped[(model, epsilon)]
        stats.total_rows += 1

        if str(row.get("status", "")).strip().lower() != "ok":
            stats.error_rows += 1
            continue

        stats.ok_rows += 1
        if truthy(row.get("class_preserved")):
            stats.class_preserved_rows += 1

        for metric in METRICS:
            value = safe_float(row.get(metric))
            if value is not None:
                stats.values[metric].append(value)
    return grouped


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else math.nan


def pstdev(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0 if values else math.nan


def build_summary_rows(
    grouped: Dict[Tuple[str, float], GroupStats],
    component: str,
    alpha_percentile: float,
) -> List[dict]:
    summary_rows: List[dict] = []
    for (model, epsilon), stats in sorted(grouped.items(), key=lambda item: (model_sort_key(item[0][0]), item[0][1])):
        row = {
            "model": model,
            "epsilon": f"{epsilon:.10g}",
            "epsilon_label": epsilon_label(epsilon),
            "component": component,
            "alpha_percentile": f"{alpha_percentile:g}",
            "total_rows": stats.total_rows,
            "ok_rows": stats.ok_rows,
            "error_rows": stats.error_rows,
            "class_preserved_rows": stats.class_preserved_rows,
            "class_preserved_rate": (
                f"{stats.class_preserved_rows / stats.ok_rows:.10g}" if stats.ok_rows else "nan"
            ),
        }

        for metric in METRICS:
            values = stats.values.get(metric, [])
            row[f"{metric}_mean"] = f"{mean(values):.10g}" if values else "nan"
            row[f"{metric}_median"] = f"{median(values):.10g}" if values else "nan"
            row[f"{metric}_stdev"] = f"{pstdev(values):.10g}" if values else "nan"
        summary_rows.append(row)
    return summary_rows


def write_summary_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError("No summary rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def values_by_model(summary_rows: Sequence[dict], y_field: str) -> Dict[str, List[Tuple[float, float]]]:
    data: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for row in summary_rows:
        epsilon = safe_float(row.get("epsilon"))
        value = safe_float(row.get(y_field))
        if epsilon is None or value is None:
            continue
        data[str(row["model"])].append((epsilon, value))
    for values in data.values():
        values.sort()
    return data


def all_epsilons(summary_rows: Sequence[dict]) -> List[float]:
    return sorted(
        {
            epsilon
            for row in summary_rows
            if (epsilon := safe_float(row.get("epsilon"))) is not None
        },
    )


def import_plotting():
    matplotlib_config_dir = Path(tempfile.gettempdir()) / "matplotlib-whitebox-sweep"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    return plt, np


def save_line_chart(
    summary_rows: Sequence[dict],
    y_field: str,
    ylabel: str,
    title: str,
    output_path: Path,
    y_limits: Optional[Tuple[float, float]] = None,
) -> bool:
    data = values_by_model(summary_rows, y_field)
    if not data:
        print(f"Skipping {output_path.name}: no values for {y_field}.")
        return False

    plt, _ = import_plotting()
    eps_values = all_epsilons(summary_rows)
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for model in sorted(data, key=model_sort_key):
        xs = [point[0] for point in data[model]]
        ys = [point[1] for point in data[model]]
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, label=model)

    if eps_values and all(value > 0 for value in eps_values):
        ax.set_xscale("log", base=2)
        ax.set_xticks(eps_values)
        ax.set_xticklabels([epsilon_label(value) for value in eps_values])
    ax.set_xlabel("Attack epsilon")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def save_heatmap(
    summary_rows: Sequence[dict],
    y_field: str,
    title: str,
    output_path: Path,
) -> bool:
    models = sorted({str(row["model"]) for row in summary_rows}, key=model_sort_key)
    eps_values = all_epsilons(summary_rows)
    if not models or not eps_values:
        print(f"Skipping {output_path.name}: no models or epsilons.")
        return False

    plt, np = import_plotting()
    model_idx = {model: idx for idx, model in enumerate(models)}
    eps_idx = {epsilon: idx for idx, epsilon in enumerate(eps_values)}
    matrix = np.full((len(models), len(eps_values)), np.nan, dtype=float)

    for row in summary_rows:
        epsilon = safe_float(row.get("epsilon"))
        value = safe_float(row.get(y_field))
        if epsilon is None or value is None:
            continue
        matrix[model_idx[str(row["model"])], eps_idx[epsilon]] = value

    if np.isnan(matrix).all():
        print(f"Skipping {output_path.name}: no values for {y_field}.")
        return False

    fig_width = max(8.0, 1.2 * len(eps_values) + 3.0)
    fig_height = max(4.2, 0.48 * len(models) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, aspect="auto", cmap="viridis")

    ax.set_xticks(range(len(eps_values)))
    ax.set_xticklabels([epsilon_label(value) for value in eps_values])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel("Attack epsilon")
    ax.set_title(title)

    for row_idx, model in enumerate(models):
        for col_idx, _epsilon in enumerate(eps_values):
            value = matrix[row_idx, col_idx]
            if math.isfinite(float(value)):
                ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=8, color="white")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(y_field.replace("_", " "))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def save_charts(summary_rows: Sequence[dict], output_dir: Path, primary_metric: str, title_prefix: str) -> List[Path]:
    paths: List[Path] = []

    chart_specs = [
        (
            f"{primary_metric}_mean",
            primary_metric.replace("_", " ").title(),
            f"{title_prefix}: {primary_metric.replace('_', ' ')}",
            output_dir / f"{primary_metric}_vs_epsilon.png",
            (-1.0, 1.0) if "spearman" in primary_metric else None,
        ),
        (
            "method_spearman_mean",
            "Method Spearman",
            f"{title_prefix}: built-in vs whitebox Spearman",
            output_dir / "method_spearman_vs_epsilon.png",
            (-1.0, 1.0),
        ),
        (
            "class_preserved_rate",
            "Class Preserved Rate",
            f"{title_prefix}: class preservation",
            output_dir / "class_preserved_rate_vs_epsilon.png",
            (0.0, 1.05),
        ),
        (
            "attack_explain_cosine_distance_mean",
            "Attack Explain Cosine Distance",
            f"{title_prefix}: cosine distance",
            output_dir / "attack_explain_cosine_distance_vs_epsilon.png",
            None,
        ),
        (
            "mean_abs_diff_mean",
            "Mean Absolute Difference",
            f"{title_prefix}: mean absolute difference",
            output_dir / "mean_abs_diff_vs_epsilon.png",
            None,
        ),
    ]

    for y_field, ylabel, title, path, y_limits in chart_specs:
        if save_line_chart(summary_rows, y_field, ylabel, title, path, y_limits=y_limits):
            paths.append(path)

    heatmap_path = output_dir / f"{primary_metric}_heatmap.png"
    if save_heatmap(
        summary_rows,
        f"{primary_metric}_mean",
        f"{title_prefix}: {primary_metric.replace('_', ' ')} heatmap",
        heatmap_path,
    ):
        paths.append(heatmap_path)

    return paths


def main() -> None:
    args = parse_args()
    result_root = args.result_root
    output_dir = args.output_dir or result_root / "charts"

    result_files, source_kind = discover_result_files(result_root)
    if not result_files:
        raise SystemExit(f"No results.csv files found under {result_root}.")

    rows = read_rows(result_files)
    if not rows:
        raise SystemExit(f"Result files under {result_root} did not contain any rows.")

    component = choose_filter_value(rows, "component", args.component)
    alpha_percentile = choose_alpha(rows, args.alpha_percentile)
    filtered_rows = [
        row
        for row in rows
        if str(row.get("component", "")).strip() == component
        and (alpha := safe_float(row.get("alpha_percentile"))) is not None
        and abs(alpha - alpha_percentile) < 1e-9
    ]
    if not filtered_rows:
        raise SystemExit("No rows left after component/alpha filtering.")

    grouped = aggregate(filtered_rows)
    summary_rows = build_summary_rows(grouped, component, alpha_percentile)
    summary_csv = output_dir / "summary_by_model_epsilon.csv"
    write_summary_csv(summary_csv, summary_rows)

    chart_paths = save_charts(summary_rows, output_dir, args.primary_metric, args.title_prefix)

    print("=" * 72)
    print("Whitebox Spearman charts")
    print("=" * 72)
    print(f"Result root : {result_root}")
    print(f"Source      : {source_kind}")
    print(f"CSV files   : {len(result_files)} selected")
    print(f"Rows        : {len(filtered_rows)} after filtering")
    print(f"Component   : {component}")
    print(f"Alpha       : {alpha_percentile:g}")
    print(f"Summary CSV : {summary_csv}")
    for path in chart_paths:
        print(f"Chart       : {path}")


if __name__ == "__main__":
    main()
