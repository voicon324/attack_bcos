#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "attacks" / "explain_guided_pixel_es_patch.py"
DEFAULT_CSV = ROOT / "data" / "used_images_500_local.csv"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "outputs" / "result" / "explain_guided_pixel_es_outputs_csv"


def validate_csv(csv_path: Path) -> None:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "image_path" not in fieldnames:
            raise ValueError(f"CSV must contain an image_path column: {csv_path}")


def build_command(args: argparse.Namespace, extra_args: List[str]) -> List[str]:
    command = [
        args.python,
        "-u",
        str(TARGET),
        "--images-csv",
        str(args.images_csv),
        "--model",
        args.model,
        "--device",
        args.device,
        "--output-dir",
        str(args.output_dir),
        "--patch-size",
        str(args.patch_size),
        "--generations",
        str(args.generations),
        "--population",
        str(args.population),
        "--elite-fraction",
        str(args.elite_fraction),
        "--sigma",
        str(args.sigma),
        "--sigma-decay",
        str(args.sigma_decay),
        "--sigma-min",
        str(args.sigma_min),
        "--lr",
        str(args.lr),
        "--epsilon",
        str(args.epsilon),
        "--norm",
        args.norm,
        "--position-rule",
        args.position_rule,
        "--image-batch-size",
        str(args.image_batch_size),
        "--score-batch-size",
        str(args.score_batch_size),
        "--save",
        args.save,
    ]

    if args.attack_original_model:
        command.append("--attack-original-model")
    if args.save_images:
        command.append("--save-images")
    if args.save_figure:
        command.append("--save-figure")
    if args.verbose_generations:
        command.append("--verbose-generations")

    return command + extra_args


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Run explain_guided_pixel_es_patch.py on images listed in a CSV.",
    )
    parser.add_argument(
        "--images-csv",
        type=Path,
        default=DEFAULT_CSV,
        help="CSV path with an image_path column.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for per-image results and summary.csv.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run the attack script.")
    parser.add_argument("--model", default="resnet50", help="B-cos model name.")
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto.")
    parser.add_argument("--patch-size", type=int, default=16, help="Square patch size.")
    parser.add_argument("--generations", "--gen", type=int, default=200, help="Number of ES generations.")
    parser.add_argument("--population", "--pop", type=int, default=50, help="ES population size.")
    parser.add_argument("--elite-fraction", type=float, default=0.2, help="Elite fraction.")
    parser.add_argument("--sigma", type=float, default=0.08, help="Initial full-patch noise std.")
    parser.add_argument("--sigma-decay", type=float, default=0.995, help="Sigma decay.")
    parser.add_argument("--sigma-min", type=float, default=0.01, help="Minimum sigma.")
    parser.add_argument("--lr", type=float, default=0.8, help="ES update rate.")
    parser.add_argument("--epsilon", type=float, default=0.3, help="Perturbation budget.")
    parser.add_argument("--norm", default="linf", choices=("l0", "l2", "linf"), help="Perturbation norm budget.")
    parser.add_argument(
        "--position-rule",
        default="margin",
        choices=("margin", "top1", "dynamic-margin", "dynamic", "random", "gradcam"),
        help="Patch position rule: current margin rule, top1, 6-channel dynamic-weight variants, random, or gradcam.",
    )
    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=128,
        help="Images attacked together in CSV mode. Use 0 for attack script auto mode.",
    )
    parser.add_argument(
        "--score-batch-size",
        type=int,
        default=0,
        help="Total candidate evaluations per score forward. 0 = auto chunking.",
    )
    parser.add_argument("--attack-original-model", action="store_true", help="Optimize/evaluate on torchvision model.")
    parser.add_argument("--save-images", action="store_true", help="Save before/after PNGs.")
    parser.add_argument("--save-figure", action="store_true", help="Save explanation figure.")
    parser.add_argument("--verbose-generations", action="store_true", help="Print one line per ES generation.")
    parser.add_argument("--save", default="explain_guided_es_patch.png", help="Output figure filename.")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running it.")
    return parser.parse_known_args()


def main() -> int:
    args, extra_args = parse_args()
    args.images_csv = args.images_csv.resolve()
    args.output_dir = args.output_dir.resolve()

    if not TARGET.is_file():
        raise FileNotFoundError(f"Attack script not found: {TARGET}")
    validate_csv(args.images_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    command = build_command(args, extra_args)
    print("Running:")
    print(" ".join(shlex.quote(part) for part in command))

    if args.dry_run:
        return 0

    completed = subprocess.run(command, cwd=str(ROOT), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
