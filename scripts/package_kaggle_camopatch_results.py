#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aggregate_kaggle_camopatch_results import aggregate


DEFAULT_RUN_ROOT = Path("kaggle_runs_success_query_full")
DEFAULT_JOBS_CONFIG = Path("kaggle/camopatch_jobs.json")


def load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_job_fields(job_id: str, job_config: dict[str, Any] | None = None) -> dict[str, Any]:
    job_config = job_config or {}
    rest = job_id.removeprefix("camopatch-bcos-")
    model = rest.split("-s", 1)[0]
    patch_size = job_config.get("patch_size", "")
    linf = job_config.get("linf", "")
    position = job_config.get("position", "")
    if not patch_size:
        marker = "-s"
        if marker in rest:
            patch_size = rest.split(marker, 1)[1].split("-", 1)[0]
    if not linf:
        marker = "-linf"
        if marker in rest:
            linf = rest.split(marker, 1)[1].split("-", 1)[0].replace("_", "/")
    if not position:
        if job_id.endswith("-bcos_top1"):
            position = "bcos_top1"
        elif job_id.endswith("-gradcam"):
            position = "gradcam"
        elif job_id.endswith("-random"):
            position = "random"
    return {
        "model": model,
        "patch_size": patch_size,
        "linf": linf,
        "position": position,
    }


def write_status_csv(run_root: Path, jobs_config: Path) -> Path:
    state = load_json(run_root / "state.json", {"jobs": {}})
    jobs_data = load_json(jobs_config, {"jobs": []})
    jobs_by_id = {str(job["job_id"]): job for job in jobs_data.get("jobs", [])}
    rows: list[dict[str, Any]] = []
    for job_id, state_job in sorted(state.get("jobs", {}).items()):
        config = jobs_by_id.get(job_id, {}).get("job_config", {})
        parsed = parse_job_fields(job_id, config)
        rows.append(
            {
                "job_id": job_id,
                "status": state_job.get("status", ""),
                "account": state_job.get("account", ""),
                "model": parsed["model"],
                "patch_size": parsed["patch_size"],
                "linf": parsed["linf"],
                "position": parsed["position"],
                "last_checked": state_job.get("last_checked", ""),
                "url": state_job.get("url", ""),
                "result_zip": state_job.get("result_zip", ""),
                "failure_reason": state_job.get("failure_reason", ""),
            }
        )
    path = run_root / "status_current.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "job_id",
        "status",
        "account",
        "model",
        "patch_size",
        "linf",
        "position",
        "last_checked",
        "url",
        "result_zip",
        "failure_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def build_manifest(run_root: Path, aggregate_dir: Path, jobs_config: Path) -> dict[str, Any]:
    state = load_json(run_root / "state.json", {"jobs": {}})
    jobs = state.get("jobs", {})
    status_counts = Counter(str(job.get("status", "")) for job in jobs.values())
    job_summary = aggregate_dir / "camopatch_job_summary.csv"
    summary = aggregate_dir / "camopatch_summary.csv"
    success_events = aggregate_dir / "camopatch_success_events.csv"
    success_by_query = aggregate_dir / "camopatch_success_by_query.csv"
    failures = load_json(aggregate_dir / "camopatch_aggregate_failures.json", [])
    return {
        "generated_at": now_iso(),
        "run_root": str(run_root),
        "jobs_config": str(jobs_config),
        "status_counts": dict(sorted(status_counts.items())),
        "total_jobs": len(jobs),
        "all_jobs_terminal": all(
            str(job.get("status", "")) in {"done", "failed"} for job in jobs.values()
        ),
        "all_jobs_done": bool(jobs)
        and all(str(job.get("status", "")) == "done" for job in jobs.values()),
        "summary_rows": count_csv_rows(summary),
        "success_event_rows": count_csv_rows(success_events),
        "success_by_query_rows": count_csv_rows(success_by_query),
        "job_summary_rows": count_csv_rows(job_summary),
        "aggregate_failure_count": len(failures),
        "files": [
            "manifest.json",
            "status_current.csv",
            "camopatch_summary.csv",
            "camopatch_success_events.csv",
            "camopatch_success_by_query.csv",
            "camopatch_job_summary.csv",
            "camopatch_aggregate_failures.json",
        ],
    }


def write_results_zip(aggregate_dir: Path, status_csv: Path, manifest: dict[str, Any], zip_path: Path) -> Path:
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = aggregate_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    files = [
        (manifest_path, "manifest.json"),
        (status_csv, "status_current.csv"),
        (aggregate_dir / "camopatch_summary.csv", "camopatch_summary.csv"),
        (aggregate_dir / "camopatch_success_events.csv", "camopatch_success_events.csv"),
        (aggregate_dir / "camopatch_success_by_query.csv", "camopatch_success_by_query.csv"),
        (aggregate_dir / "camopatch_job_summary.csv", "camopatch_job_summary.csv"),
        (aggregate_dir / "camopatch_aggregate_failures.json", "camopatch_aggregate_failures.json"),
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, arcname in files:
            if path.is_file():
                archive.write(path, arcname)
    with zipfile.ZipFile(zip_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Invalid result zip member: {bad_member}")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate and zip Kaggle CamoPatch result CSVs for download."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--jobs-config", type=Path, default=DEFAULT_JOBS_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--zip-path", type=Path, default=None)
    args = parser.parse_args()

    aggregate_dir = args.output_dir or args.run_root / "aggregate"
    aggregate(args.run_root, args.jobs_config, aggregate_dir)
    status_csv = write_status_csv(args.run_root, args.jobs_config)
    manifest = build_manifest(args.run_root, aggregate_dir, args.jobs_config)
    zip_path = args.zip_path or aggregate_dir / "camopatch_results_only_latest.zip"
    write_results_zip(aggregate_dir, status_csv, manifest, zip_path)
    print(f"zip_path={zip_path}")
    print(f"status_counts={manifest['status_counts']}")
    print(f"summary_rows={manifest['summary_rows']}")
    print(f"aggregate_failure_count={manifest['aggregate_failure_count']}")


if __name__ == "__main__":
    main()
