#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_STATUSES = {"queued", "staging", "submitting", "running", "downloading"}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_state(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def status_counts(state: dict) -> Counter[str]:
    return Counter(str(job.get("status", "")) for job in state.get("jobs", {}).values())


def is_terminal(counts: Counter[str]) -> bool:
    return not any(counts.get(status, 0) for status in ACTIVE_STATUSES)


def run_logged(cmd: list[str], log_path: Path, cwd: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{now_iso()}] run {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True)
        log.write(f"[{now_iso()}] rc={proc.returncode}\n")
        log.flush()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def python_cmd(conda_env: str | None) -> list[str]:
    if conda_env:
        return ["conda", "run", "-n", conda_env, "python"]
    return [sys.executable]


def zip_paths(zip_path: Path, paths: list[Path]) -> None:
    if zip_path.exists():
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for root_path in paths:
            if not root_path.exists():
                continue
            if root_path.is_file():
                archive.write(root_path, root_path.relative_to(zip_path.parent))
                continue
            for path in sorted(root_path.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(zip_path.parent))


def plot_all(args: argparse.Namespace) -> None:
    env = dict(**__import__("os").environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    py = python_cmd(args.conda_env)
    aggregate_cmd = [
        *py,
        "scripts/aggregate_camopatch_all_results.py",
        "--output-dir",
        str(args.output_dir),
        "--clean-predictions-csv",
        str(args.clean_predictions_csv),
    ]
    bar_cmd = [
        *py,
        "scripts/plot_camopatch_position_bar_images.py",
        "--summary-dir",
        str(args.output_dir),
        "--output-dir",
        str(args.output_dir / "charts_position_bars"),
        "--attack",
        "camopatch",
    ]
    movable_curve_cmd = [
        *py,
        "scripts/plot_camopatch_movable_query_curves.py",
        "--summary-dir",
        str(args.output_dir),
        "--output-dir",
        str(args.output_dir / "charts_movable_query_curves"),
        "--position-mode",
        "movable",
        "--attack",
        "camopatch",
    ]
    fixed_curve_cmd = [
        *py,
        "scripts/plot_camopatch_movable_query_curves.py",
        "--summary-dir",
        str(args.output_dir),
        "--output-dir",
        str(args.output_dir / "charts_fixed_query_curves"),
        "--position-mode",
        "fixed",
        "--attack",
        "camopatch",
    ]

    for cmd in (aggregate_cmd, bar_cmd, movable_curve_cmd, fixed_curve_cmd):
        with args.log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{now_iso()}] run {' '.join(cmd)}\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
            log.write(f"[{now_iso()}] rc={proc.returncode}\n")
            log.flush()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

    zip_paths(
        args.output_dir / "camopatch_charts_latest.zip",
        [
            args.output_dir / "charts_position_bars",
            args.output_dir / "charts_fixed_query_curves",
            args.output_dir / "charts_movable_query_curves",
        ],
    )
    zip_paths(
        args.output_dir / "camopatch_analysis_latest.zip",
        [
            args.output_dir / "combined_image_results_all.csv",
            args.output_dir / "combined_image_results_clean_correct.csv",
            args.output_dir / "summary_all_images.csv",
            args.output_dir / "summary_clean_correct.csv",
            args.output_dir / "success_by_query_all_images.csv",
            args.output_dir / "success_by_query_clean_correct.csv",
            args.output_dir / "included_jobs.csv",
            args.output_dir / "skipped_jobs.csv",
            args.output_dir / "manifest.json",
            args.output_dir / "charts_position_bars",
            args.output_dir / "charts_fixed_query_curves",
            args.output_dir / "charts_movable_query_curves",
        ],
    )
    with args.log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{now_iso()}] wrote {args.output_dir / 'camopatch_charts_latest.zip'}\n")
        log.write(f"[{now_iso()}] wrote {args.output_dir / 'camopatch_analysis_latest.zip'}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for CamoPatch Kaggle jobs, then regenerate aggregate charts.")
    parser.add_argument("--run-root", type=Path, default=ROOT / "kaggle_runs_success_query_full")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "analysis" / "camopatch_all_results_latest")
    parser.add_argument(
        "--clean-predictions-csv",
        type=Path,
        default=ROOT / "artifacts" / "analysis" / "camopatch_all_results_latest" / "clean_predictions_1000.csv",
    )
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--max-wait-minutes", type=int, default=0, help="0 means wait forever.")
    parser.add_argument("--conda-env", default="bcos")
    parser.add_argument("--log-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_root = args.run_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.clean_predictions_csv = args.clean_predictions_csv.resolve()
    args.log_path = (args.log_path or args.output_dir / "wait_and_plot.log").resolve()
    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    state_path = args.run_root / "state.json"
    started = time.monotonic()
    while True:
        state = load_state(state_path)
        counts = status_counts(state)
        with args.log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{now_iso()}] status {dict(counts)}\n")
        if is_terminal(counts):
            plot_all(args)
            return
        if args.max_wait_minutes > 0 and time.monotonic() - started > args.max_wait_minutes * 60:
            raise TimeoutError(f"Timed out waiting for terminal state: {dict(counts)}")
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    main()
