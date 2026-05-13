"""
Summarize explain-guided ES patch attack results from one or more summary.csv files.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


REQUIRED_COLUMNS = {
    "index",
    "image_path",
    "output_dir",
    "target_class",
    "original_logit",
    "perturbed_class",
    "perturbed_target_logit",
    "perturbed_pred_logit",
    "patch_pos_y",
    "patch_pos_x",
    "norm",
    "epsilon",
    "success",
}


def parse_success(value: str) -> int:
    key = value.strip().lower()
    if key in {"1", "true", "yes"}:
        return 1
    if key in {"0", "false", "no"}:
        return 0
    raise ValueError(f"Unsupported success value: {value!r}")


def parse_row(row: Dict[str, str], line_no: int, source: Path) -> Dict[str, Any]:
    try:
        index_text = (row.get("index") or "").strip()
        index = int(index_text) if index_text else None
        original_logit = float(row["original_logit"])
        perturbed_target_logit = float(row["perturbed_target_logit"])
        parsed = {
            "index": index,
            "image_path": row["image_path"],
            "output_dir": row["output_dir"],
            "target_class": int(row["target_class"]),
            "original_logit": original_logit,
            "perturbed_class": int(row["perturbed_class"]),
            "perturbed_target_logit": perturbed_target_logit,
            "perturbed_pred_logit": float(row["perturbed_pred_logit"]),
            "patch_pos_y": int(row["patch_pos_y"]),
            "patch_pos_x": int(row["patch_pos_x"]),
            "norm": row["norm"].strip(),
            "epsilon": row["epsilon"].strip(),
            "success": parse_success(row["success"]),
        }
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        raise ValueError(f"{source}:{line_no}: failed to parse row: {exc}") from exc

    parsed["target_logit_delta"] = perturbed_target_logit - original_logit
    parsed["class_changed"] = int(parsed["perturbed_class"] != parsed["target_class"])
    parsed["ground_truth_class"] = infer_ground_truth_class(parsed["image_path"])
    parsed["initially_correct"] = int(
        parsed["ground_truth_class"] is not None and parsed["target_class"] == parsed["ground_truth_class"]
    )
    return parsed


def infer_ground_truth_class(image_path: str) -> Optional[int]:
    parent_name = Path(image_path).parent.name.strip()
    if not parent_name.isdigit():
        return None
    return int(parent_name)


def load_summary(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - fieldnames)
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")

        rows = [parse_row(row, line_no, path) for line_no, row in enumerate(reader, start=2)]

    if not rows:
        raise ValueError(f"{path}: summary file is empty")
    return rows


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values)


def median(values: Sequence[float]) -> float:
    return statistics.median(values)


def summarize_rows(path: Path, rows: List[Dict[str, Any]], only_initially_correct: bool) -> Dict[str, Any]:
    source_total = len(rows)
    source_initially_correct = sum(row["initially_correct"] for row in rows)
    source_inferable = sum(row["ground_truth_class"] is not None for row in rows)
    if only_initially_correct:
        rows = [row for row in rows if row["initially_correct"]]
        if not rows:
            raise ValueError(
                f"{path}: no rows remain after filtering to initially correct predictions "
                "(ground truth inferred from image_path parent directory)."
            )

    total = len(rows)
    successes = sum(row["success"] for row in rows)
    changed_classes = sum(row["class_changed"] for row in rows)
    deltas = [row["target_logit_delta"] for row in rows]
    original_logits = [row["original_logit"] for row in rows]
    perturbed_target_logits = [row["perturbed_target_logit"] for row in rows]
    patch_pos_y = [row["patch_pos_y"] for row in rows]
    patch_pos_x = [row["patch_pos_x"] for row in rows]

    return {
        "path": path,
        "rows": rows,
        "total": total,
        "source_total": source_total,
        "source_initially_correct": source_initially_correct,
        "source_inferable": source_inferable,
        "filtered_initially_correct": only_initially_correct,
        "successes": successes,
        "success_rate": successes / total * 100.0,
        "changed_classes": changed_classes,
        "norms": sorted({row["norm"] for row in rows}),
        "epsilons": sorted({row["epsilon"] for row in rows}),
        "original_logit_mean": mean(original_logits),
        "perturbed_target_logit_mean": mean(perturbed_target_logits),
        "delta_mean": mean(deltas),
        "delta_median": median(deltas),
        "delta_min": min(deltas),
        "delta_max": max(deltas),
        "patch_pos_y_mean": mean(patch_pos_y),
        "patch_pos_y_min": min(patch_pos_y),
        "patch_pos_y_max": max(patch_pos_y),
        "patch_pos_x_mean": mean(patch_pos_x),
        "patch_pos_x_min": min(patch_pos_x),
        "patch_pos_x_max": max(patch_pos_x),
    }


def format_case(row: Dict[str, Any]) -> str:
    index_text = "?" if row["index"] is None else str(row["index"])
    image_name = Path(row["image_path"]).name
    return (
        f"#{index_text} {image_name} | "
        f"class {row['target_class']} -> {row['perturbed_class']} | "
        f"delta {row['target_logit_delta']:+.4f} | "
        f"patch ({row['patch_pos_y']}, {row['patch_pos_x']})"
    )


def top_k_by_delta(rows: Iterable[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda row: row["target_logit_delta"])[:top_k]


def print_single_summary(summary: Dict[str, Any], top_k: int) -> None:
    path = summary["path"]
    rows = summary["rows"]
    success_rows = [row for row in rows if row["success"]]

    print("=" * 80)
    print(f"Summary: {path}")
    print("=" * 80)
    if summary["filtered_initially_correct"]:
        print(
            f"Filtered rows         : {summary['total']}/{summary['source_total']} "
            f"(initially correct; inferable GT rows: {summary['source_inferable']})"
        )
    print(f"Rows                  : {summary['total']}")
    print(f"Norm                  : {', '.join(summary['norms'])}")
    print(f"Epsilon               : {', '.join(summary['epsilons'])}")
    print(
        f"Success               : {summary['successes']}/{summary['total']} "
        f"({summary['success_rate']:.2f}%)"
    )
    print(f"Changed classes       : {summary['changed_classes']}/{summary['total']}")
    print(
        f"Target logit mean     : {summary['original_logit_mean']:.4f} "
        f"-> {summary['perturbed_target_logit_mean']:.4f}"
    )
    print(
        f"Target logit delta    : mean {summary['delta_mean']:+.4f} | "
        f"median {summary['delta_median']:+.4f} | "
        f"min {summary['delta_min']:+.4f} | max {summary['delta_max']:+.4f}"
    )
    print(
        f"Patch pos Y           : mean {summary['patch_pos_y_mean']:.1f} | "
        f"min {summary['patch_pos_y_min']} | max {summary['patch_pos_y_max']}"
    )
    print(
        f"Patch pos X           : mean {summary['patch_pos_x_mean']:.1f} | "
        f"min {summary['patch_pos_x_min']} | max {summary['patch_pos_x_max']}"
    )

    print("\nTop target-logit drops:")
    for row in top_k_by_delta(rows, top_k):
        print(f"  {format_case(row)}")

    print("\nSuccessful attacks:")
    if success_rows:
        for row in top_k_by_delta(success_rows, min(top_k, len(success_rows))):
            print(f"  {format_case(row)}")
    else:
        print("  None")
    print()


def print_comparison_table(summaries: Sequence[Dict[str, Any]]) -> None:
    print("=" * 80)
    print("Comparison")
    print("=" * 80)
    header = (
        f"{'File':<44} {'Rows':>6} {'Succ':>6} {'Rate%':>8} "
        f"{'MeanΔ':>10} {'MedianΔ':>10} {'MinΔ':>10}"
    )
    print(header)
    print("-" * len(header))
    for summary in summaries:
        file_name = summary["path"].parent.parent.name
        if len(file_name) > 44:
            file_name = file_name[:41] + "..."
        print(
            f"{file_name:<44} {summary['total']:>6d} {summary['successes']:>6d} "
            f"{summary['success_rate']:>7.2f}% {summary['delta_mean']:>+10.4f} "
            f"{summary['delta_median']:>+10.4f} {summary['delta_min']:>+10.4f}"
        )
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize explain-guided ES patch attack results from summary.csv files.",
    )
    parser.add_argument(
        "summary_csv",
        nargs="+",
        type=Path,
        help="One or more summary.csv files to summarize.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many notable rows to print for each file.",
    )
    parser.add_argument(
        "--only-initially-correct",
        action="store_true",
        help=(
            "Only summarize rows where the initial prediction matches ground truth. "
            "Ground truth is inferred from the numeric parent directory in image_path."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be > 0")

    summaries = [
        summarize_rows(path, load_summary(path), only_initially_correct=args.only_initially_correct)
        for path in args.summary_csv
    ]

    if len(summaries) > 1:
        print_comparison_table(summaries)
    for summary in summaries:
        print_single_summary(summary, top_k=args.top_k)


if __name__ == "__main__":
    main()
