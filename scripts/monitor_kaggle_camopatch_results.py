#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def terminal(state: dict) -> bool:
    jobs = state.get("jobs", {})
    return bool(jobs) and all(
        str(job.get("status", "")) in {"done", "failed"} for job in jobs.values()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Periodically package CamoPatch Kaggle aggregate results.")
    parser.add_argument("--run-root", type=Path, default=Path("kaggle_runs_success_query_full"))
    parser.add_argument("--jobs-config", type=Path, default=Path("kaggle/camopatch_jobs.json"))
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--log-path", type=Path, default=None)
    args = parser.parse_args()

    log_path = args.log_path or args.run_root / "aggregate" / "monitor.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        cmd = [
            sys.executable,
            "scripts/package_kaggle_camopatch_results.py",
            "--run-root",
            str(args.run_root),
            "--jobs-config",
            str(args.jobs_config),
            "--output-dir",
            str(args.run_root / "aggregate"),
            "--zip-path",
            str(args.run_root / "aggregate" / "camopatch_results_only_latest.zip"),
        ]
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{now_iso()}] package start\n")
            log.flush()
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
            log.write(f"[{now_iso()}] package rc={proc.returncode}\n")
            log.flush()
        if terminal(load_json(args.run_root / "state.json")):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
