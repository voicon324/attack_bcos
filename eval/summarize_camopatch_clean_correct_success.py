"""
Compute CamoPatch attack success rate on images that were cleanly classified.

The CamoPatch summary.csv records final attack success, but not the model's
clean prediction. This script joins it with a clean-prediction CSV and reports
attack success over the clean-correct subset.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ATTACK_SUMMARY = (
    PROJECT_ROOT
    / "artifacts"
    / "outputs"
    / "camopatch_torchvision_resnet50_strict_1p1_full_linf64_256_fixed"
    / "summary.csv"
)
DEFAULT_CLEAN_CSV = (
    PROJECT_ROOT
    / "artifacts"
    / "outputs"
    / "camopatch_torchvision_resnet50_strict_1p1_full_linf64_256"
    / "clean_torchvision_resnet50_predictions_vs_attack_summary.csv"
)


def parse_bool_int(value: str, *, column: str) -> int:
    key = value.strip().lower()
    if key in {"1", "true", "yes"}:
        return 1
    if key in {"0", "false", "no"}:
        return 0
    raise ValueError(f"Unsupported boolean value in {column}: {value!r}")


def require_columns(path: Path, fieldnames: Sequence[str] | None, required: Sequence[str]) -> None:
    fields = set(fieldnames or [])
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no data rows.")
    return rows


def default_rows_csv(attack_summary_csv: Path) -> Path:
    return attack_summary_csv.parent / "clean_correct_attack_success_rows.csv"


def default_summary_csv(attack_summary_csv: Path) -> Path:
    return attack_summary_csv.parent / "success_rate_on_clean_correct.csv"


def summarize(
    attack_summary_csv: Path,
    clean_csv: Path,
    rows_csv: Path,
    summary_csv: Path,
    join_column: str,
    success_column: str,
    clean_correct_column: str,
) -> Dict[str, object]:
    attack_rows = load_rows(attack_summary_csv)
    clean_rows = load_rows(clean_csv)

    require_columns(
        attack_summary_csv,
        attack_rows[0].keys(),
        [join_column, "true_label", success_column, "final_prediction", "image_path"],
    )
    require_columns(clean_csv, clean_rows[0].keys(), [join_column, clean_correct_column])

    clean_by_key: Dict[str, Dict[str, str]] = {}
    for row in clean_rows:
        key = row[join_column].strip()
        if key in clean_by_key:
            raise ValueError(f"{clean_csv}: duplicate {join_column}={key!r}")
        clean_by_key[key] = row

    joined_rows: List[Dict[str, object]] = []
    for row in attack_rows:
        key = row[join_column].strip()
        if key not in clean_by_key:
            raise ValueError(f"{clean_csv}: missing clean row for {join_column}={key!r}")
        clean = clean_by_key[key]
        clean_correct = parse_bool_int(clean[clean_correct_column], column=clean_correct_column)
        adversarial = parse_bool_int(row[success_column], column=success_column)
        joined_rows.append(
            {
                join_column: key,
                "true_label": row["true_label"],
                "clean_prediction": clean.get("clean_prediction", ""),
                "clean_correct": clean_correct,
                success_column: adversarial,
                "final_prediction": row["final_prediction"],
                "attack_success_on_clean_correct": int(clean_correct == 1 and adversarial == 1),
                "image_path": row["image_path"],
            }
        )

    rows_csv.parent.mkdir(parents=True, exist_ok=True)
    row_fieldnames = [
        join_column,
        "true_label",
        "clean_prediction",
        "clean_correct",
        success_column,
        "final_prediction",
        "attack_success_on_clean_correct",
        "image_path",
    ]
    with rows_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fieldnames)
        writer.writeheader()
        writer.writerows(joined_rows)

    total_images = len(joined_rows)
    clean_correct_images = sum(int(row["clean_correct"]) for row in joined_rows)
    successes_on_clean_correct = sum(int(row["attack_success_on_clean_correct"]) for row in joined_rows)
    all_successes = sum(int(row[success_column]) for row in joined_rows)
    if clean_correct_images == 0:
        raise ValueError("No clean-correct rows found; cannot compute success rate.")

    summary = {
        "attack_summary_csv": str(attack_summary_csv),
        "clean_prediction_csv": str(clean_csv),
        "rows_csv": str(rows_csv),
        "total_images": total_images,
        "clean_correct_images": clean_correct_images,
        "successes_on_clean_correct_images": successes_on_clean_correct,
        "success_rate_on_clean_correct_pct": successes_on_clean_correct / clean_correct_images * 100.0,
        "all_successes": all_successes,
        "all_success_rate_pct": all_successes / total_images * 100.0,
    }

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize CamoPatch attack success over clean-correct images.",
    )
    parser.add_argument(
        "--attack-summary",
        type=Path,
        default=DEFAULT_ATTACK_SUMMARY,
        help="CamoPatch summary.csv containing adversarial/final_prediction columns.",
    )
    parser.add_argument(
        "--clean-csv",
        type=Path,
        default=DEFAULT_CLEAN_CSV,
        help="CSV containing clean_correct predictions for the same image indices.",
    )
    parser.add_argument(
        "--rows-csv",
        type=Path,
        default=None,
        help="Output per-image joined CSV. Defaults beside --attack-summary.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Output one-row summary CSV. Defaults beside --attack-summary.",
    )
    parser.add_argument(
        "--join-column",
        default="index",
        help="Column used to join attack and clean CSVs.",
    )
    parser.add_argument(
        "--success-column",
        default="adversarial",
        help="Attack-summary column where 1 means attack success.",
    )
    parser.add_argument(
        "--clean-correct-column",
        default="clean_correct",
        help="Clean CSV column where 1 means the original prediction was correct.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows_csv = args.rows_csv or default_rows_csv(args.attack_summary)
    summary_csv = args.summary_csv or default_summary_csv(args.attack_summary)

    summary = summarize(
        attack_summary_csv=args.attack_summary,
        clean_csv=args.clean_csv,
        rows_csv=rows_csv,
        summary_csv=summary_csv,
        join_column=args.join_column,
        success_column=args.success_column,
        clean_correct_column=args.clean_correct_column,
    )

    print(f"Rows CSV    : {summary['rows_csv']}")
    print(f"Summary CSV : {summary_csv}")
    print(
        "Success on clean-correct: "
        f"{summary['successes_on_clean_correct_images']}/{summary['clean_correct_images']} "
        f"({summary['success_rate_on_clean_correct_pct']:.2f}%)"
    )
    print(
        "Success on all images    : "
        f"{summary['all_successes']}/{summary['total_images']} "
        f"({summary['all_success_rate_pct']:.2f}%)"
    )


if __name__ == "__main__":
    main()
