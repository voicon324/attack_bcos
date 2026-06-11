#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_RUN_ROOT = Path("kaggle_runs")
DEFAULT_JOBS_CONFIG = Path("kaggle/camopatch_jobs.json")


def load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_zip_member(archive: zipfile.ZipFile, suffix: str) -> str | None:
    exact = suffix.lstrip("/")
    if exact in archive.namelist():
        return exact
    matches = sorted(name for name in archive.namelist() if name.endswith(suffix))
    return matches[0] if matches else None


def read_manifest(archive: zipfile.ZipFile) -> dict[str, Any]:
    name = find_zip_member(archive, "manifest.json")
    if name is None:
        return {}
    with archive.open(name) as handle:
        return json.loads(handle.read().decode("utf-8"))


def read_summary_rows(archive: zipfile.ZipFile) -> list[dict[str, str]]:
    return read_csv_rows(archive, "outputs/summary.csv", "summary.csv")


def read_success_event_rows(archive: zipfile.ZipFile) -> list[dict[str, str]]:
    return read_csv_rows(archive, "outputs/success_events.csv", "success_events.csv")


def read_success_by_query_rows(archive: zipfile.ZipFile) -> list[dict[str, str]]:
    return read_csv_rows(archive, "outputs/success_by_query.csv", "success_by_query.csv")


def read_csv_rows(archive: zipfile.ZipFile, *suffixes: str) -> list[dict[str, str]]:
    name = None
    for suffix in suffixes:
        name = find_zip_member(archive, suffix)
        if name is not None:
            break
    if name is None:
        return []
    with archive.open(name) as handle:
        text = io.TextIOWrapper(handle, encoding="utf-8", newline="")
        return list(csv.DictReader(text))


def preferred_job_config(job_id: str, manifest: dict[str, Any], jobs_by_id: dict[str, dict]) -> dict[str, Any]:
    manifest_config = manifest.get("job_config")
    if isinstance(manifest_config, dict) and manifest_config:
        return manifest_config
    return dict(jobs_by_id.get(job_id, {}).get("job_config", {}))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(run_root: Path, jobs_config: Path, output_dir: Path) -> tuple[Path, Path, Path, Path, Path]:
    state = load_json(run_root / "state.json", {"jobs": {}})
    jobs_data = load_json(jobs_config, {"jobs": []})
    jobs_by_id = {str(job["job_id"]): job for job in jobs_data.get("jobs", [])}

    detail_rows: list[dict[str, Any]] = []
    success_event_rows: list[dict[str, Any]] = []
    success_by_query_rows: list[dict[str, Any]] = []
    job_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for job_id in sorted(state.get("jobs", {})):
        state_job = state["jobs"][job_id]
        status = str(state_job.get("status", ""))
        zip_value = str(state_job.get("result_zip", ""))
        zip_path = Path(zip_value) if zip_value else Path()
        config = dict(jobs_by_id.get(job_id, {}).get("job_config", {}))
        job_status = {
            "job_id": job_id,
            "account": state_job.get("account", ""),
            "status": status,
            "url": state_job.get("url", ""),
            "result_zip": zip_value,
            "failure_reason": state_job.get("failure_reason", ""),
            "last_checked": state_job.get("last_checked", ""),
            "done_at": state_job.get("done_at", ""),
            "rows": 0,
            "adversarial": 0,
            "job_model": config.get("model", ""),
            "job_patch_size": config.get("patch_size", ""),
            "job_linf": config.get("linf", ""),
            "job_position": config.get("position", ""),
            "job_queries": config.get("queries", ""),
            "elapsed_sec": "",
            "return_code": "",
        }

        if status != "done":
            job_rows.append(job_status)
            continue
        if not zip_path.is_file():
            job_status["failure_reason"] = "done result zip missing locally"
            failures.append(dict(job_status))
            job_rows.append(job_status)
            continue

        try:
            with zipfile.ZipFile(zip_path) as archive:
                bad_member = archive.testzip()
                if bad_member is not None:
                    job_status["failure_reason"] = f"invalid zip member: {bad_member}"
                    failures.append(dict(job_status))
                    job_rows.append(job_status)
                    continue
                manifest = read_manifest(archive)
                rows = read_summary_rows(archive)
                has_success_events = find_zip_member(archive, "outputs/success_events.csv") is not None
                has_success_by_query = find_zip_member(archive, "outputs/success_by_query.csv") is not None
                events = read_success_event_rows(archive)
                by_query = read_success_by_query_rows(archive)
        except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
            job_status["failure_reason"] = f"zip read error: {exc}"
            failures.append(dict(job_status))
            job_rows.append(job_status)
            continue

        if rows and "first_success_query" not in rows[0]:
            job_status["failure_reason"] = "summary missing first_success_query"
            failures.append(dict(job_status))
            job_rows.append(job_status)
            continue
        if not has_success_events:
            job_status["failure_reason"] = "missing success_events.csv"
            failures.append(dict(job_status))
            job_rows.append(job_status)
            continue
        if not has_success_by_query:
            job_status["failure_reason"] = "missing success_by_query.csv"
            failures.append(dict(job_status))
            job_rows.append(job_status)
            continue

        config = preferred_job_config(job_id, manifest, jobs_by_id)
        metadata = {
            "job_id": job_id,
            "account": state_job.get("account", ""),
            "url": state_job.get("url", ""),
            "result_zip": zip_value,
            "job_model": config.get("model", ""),
            "job_patch_size": config.get("patch_size", ""),
            "job_linf": config.get("linf", ""),
            "job_position": config.get("position", ""),
            "job_queries": config.get("queries", ""),
            "job_images_csv": config.get("images_csv", ""),
            "elapsed_sec": manifest.get("elapsed_sec", ""),
            "return_code": manifest.get("return_code", ""),
        }
        for row in rows:
            detail_rows.append({**metadata, **row})
        for row in events:
            success_event_rows.append({**metadata, **row})
        for row in by_query:
            success_by_query_rows.append({**metadata, **row})

        job_status["rows"] = len(rows)
        job_status["adversarial"] = sum(int(row.get("adversarial", 0) or 0) for row in rows)
        job_status.update(
            {
                "job_model": config.get("model", ""),
                "job_patch_size": config.get("patch_size", ""),
                "job_linf": config.get("linf", ""),
                "job_position": config.get("position", ""),
                "job_queries": config.get("queries", ""),
                "elapsed_sec": manifest.get("elapsed_sec", ""),
                "return_code": manifest.get("return_code", ""),
            }
        )
        job_rows.append(job_status)

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "camopatch_summary.csv"
    success_events_path = output_dir / "camopatch_success_events.csv"
    success_by_query_path = output_dir / "camopatch_success_by_query.csv"
    job_path = output_dir / "camopatch_job_summary.csv"
    failures_path = output_dir / "camopatch_aggregate_failures.json"
    write_csv(detail_path, detail_rows)
    write_csv(success_events_path, success_event_rows)
    write_csv(success_by_query_path, success_by_query_rows)
    write_csv(job_path, job_rows)
    failures_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    return detail_path, success_events_path, success_by_query_path, job_path, failures_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate downloaded CamoPatch Kaggle result zips.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--jobs-config", type=Path, default=DEFAULT_JOBS_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN_ROOT / "aggregate")
    args = parser.parse_args()

    detail_path, success_events_path, success_by_query_path, job_path, failures_path = aggregate(
        args.run_root,
        args.jobs_config,
        args.output_dir,
    )
    print(f"summary_csv={detail_path}")
    print(f"success_events_csv={success_events_path}")
    print(f"success_by_query_csv={success_by_query_path}")
    print(f"job_summary_csv={job_path}")
    print(f"failures_json={failures_path}")


if __name__ == "__main__":
    main()
