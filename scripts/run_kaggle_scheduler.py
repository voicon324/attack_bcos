#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


ACTIVE_STATUSES = {"submitting", "running", "downloading"}
TERMINAL_STATUSES = {"done", "failed"}
GPU_CAPACITY_PATTERNS = (
    "Maximum batch GPU session count",
    "Maximum simultaneous GPU session count",
)
GPU_QUOTA_PATTERNS = (
    "Maximum weekly GPU quota",
    "GPU quota of",
)
COMPETITION_RULES_PATTERNS = (
    "must accept this competition's rules",
    "accept the competition rules",
)
DEFAULT_DATASET_SOURCES = [
    "hkhnhduy/attack-bcos-github",
    "hkhnhduy/weights-bcos",
    "sautkin/imagenet1kvalid",
]
DEFAULT_BUNDLE_ESTIMATE_HOURS = {
    "resnet18": 1.9,
    "resnet50": 3.1,
    "densenet121": 3.2,
    "convnext_tiny": 3.3,
    "convnext_base": 3.8,
    "vitc_s": 2.7,
    "vitc_b": 3.6,
}
DEFAULT_BUNDLE_ESTIMATE_HOURS_BY_MODEL_PATCH = {
    ("resnet18", 8): 1.9,
    ("resnet18", 16): 1.9,
    ("resnet18", 32): 2.6,
    ("resnet50", 8): 3.2,
    ("resnet50", 16): 3.2,
    ("resnet50", 32): 3.8,
    ("densenet121", 8): 3.4,
    ("densenet121", 16): 3.4,
    ("densenet121", 32): 7.0,
    ("convnext_tiny", 8): 3.4,
    ("convnext_tiny", 16): 3.4,
    ("convnext_tiny", 32): 6.0,
    ("convnext_base", 8): 5.8,
    ("convnext_base", 16): 5.6,
    ("convnext_base", 32): 7.0,
    ("vitc_s", 8): 4.5,
    ("vitc_s", 16): 2.8,
    ("vitc_s", 32): 4.5,
    ("vitc_b", 8): 4.8,
    ("vitc_b", 16): 4.7,
    ("vitc_b", 32): 5.2,
}
DEFAULT_WEEKLY_GPU_QUOTA_HOURS = 30.0
DEFAULT_QUOTA_RESET_WEEKDAY = 5  # Saturday, Python weekday numbering.
DEFAULT_QUOTA_RESET_HOUR = 0
DEFAULT_QUOTA_RESET_TIMEZONE = "UTC"
DEFAULT_AUTO_BUNDLE_UNDER_QUOTA_HOURS = 4.0
DEFAULT_AUTO_BUNDLE_TARGET_HOURS = 7.5
DEFAULT_BUNDLE_TIMEOUT_MINUTES = 840


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    tmp_path.replace(path)


def append_progress(run_root: Path, message: str) -> None:
    path = run_root / "progress.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{now_iso()} {message}\n")


def sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "job"


def short_slug(job_id: str) -> str:
    slug = sanitize_slug(job_id)
    slug = slug.replace("camopatch-bcos-", "cb-")
    # Kaggle rejects long kernel slugs with a generic 400. Leave room for
    # "-<timestamp>" appended by stage_kernel.
    return slug[:39].strip("-")


def read_kaggle_username(kaggle_json: Path) -> str:
    data = load_json(kaggle_json, {})
    username = str(data.get("username", "")).strip()
    if not username:
        raise ValueError(f"{kaggle_json} does not contain a username.")
    return username


def install_account_config(kaggle_json: Path, runtime_dir: Path) -> Path:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    dest = runtime_dir / "kaggle.json"
    shutil.copy2(kaggle_json, dest)
    os.chmod(dest, 0o600)
    return runtime_dir


def discover_accounts(run_root: Path, accounts_config: Path | None) -> list[dict]:
    accounts: list[dict] = []
    if accounts_config and accounts_config.is_file():
        data = load_json(accounts_config, {})
        accounts.extend(data.get("accounts", []))
    else:
        main = Path("artifacts/secrets/kaggle.json")
        if main.is_file():
            accounts.append({"name": "main", "kaggle_json": str(main), "max_running": 2})
        for path in sorted((run_root / "accounts").glob("*/kaggle.json")):
            accounts.append({"name": path.parent.name, "kaggle_json": str(path), "max_running": 2})
        for path in sorted(Path("artifacts/secrets").glob("kaggle_*.json")):
            accounts.append(
                {
                    "name": path.stem.removeprefix("kaggle_"),
                    "kaggle_json": str(path),
                    "max_running": 2,
                }
            )
        for path in sorted(Path("artifacts/secrets/kaggle_accounts").glob("*/kaggle.json")):
            accounts.append({"name": path.parent.name, "kaggle_json": str(path), "max_running": 2})

    normalized: list[dict] = []
    seen_paths: set[str] = set()
    for account in accounts:
        kaggle_json = Path(account["kaggle_json"])
        if not kaggle_json.is_file():
            continue
        real_path = str(kaggle_json.resolve())
        if real_path in seen_paths:
            continue
        seen_paths.add(real_path)
        name = sanitize_slug(str(account.get("name") or kaggle_json.parent.name))
        normalized_account = {
            "name": name,
            "kaggle_json": str(kaggle_json),
            "max_running": int(account.get("max_running", 2)),
            "username": read_kaggle_username(kaggle_json),
        }
        for key in (
            "weekly_gpu_quota_hours",
            "quota_reset_weekday",
            "quota_reset_hour",
            "quota_reset_timezone",
            "auto_bundle_under_quota_hours",
            "auto_bundle_target_hours",
            "bundle_max_jobs",
        ):
            if key in account:
                normalized_account[key] = account[key]
        normalized.append(normalized_account)
    if not normalized:
        raise FileNotFoundError(
            "No Kaggle accounts found. Add artifacts/secrets/kaggle.json or "
            "kaggle_runs/accounts/<name>/kaggle.json."
        )
    return normalized


def load_jobs(path: Path) -> dict[str, dict]:
    data = load_json(path, {})
    jobs = data.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError(f"{path} must contain a top-level jobs list.")
    return {str(job["job_id"]): job for job in jobs}


def ensure_jobs_in_state(state: dict, jobs_by_id: dict[str, dict], run_root: Path) -> None:
    state.setdefault("jobs", {})
    for job_id, job in jobs_by_id.items():
        if job_id in state["jobs"]:
            continue
        state["jobs"][job_id] = {
            "job_id": job_id,
            "status": "queued",
            "account": "",
            "url": "",
            "slug": "",
            "expected_zip": job.get("expected_zip", f"{job_id}_result.zip"),
            "created_at": now_iso(),
            "last_checked": "",
            "tries": 0,
        }
        append_progress(run_root, f"queued job={job_id}")


def write_dashboard(run_root: Path, state: dict) -> None:
    path = run_root / "dashboard.tsv"
    rows = ["job_id\taccount\tstatus\turl\texpected_zip\tlast_checked"]
    for job_id in sorted(state.get("jobs", {})):
        job = state["jobs"][job_id]
        rows.append(
            "\t".join(
                [
                    job_id,
                    str(job.get("account", "")),
                    str(job.get("status", "")),
                    str(job.get("url", "")),
                    str(job.get("expected_zip", "")),
                    str(job.get("last_checked", "")),
                ]
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def run_kaggle(args: list[str], account: dict, run_root: Path, log_path: Path) -> int:
    runtime_config_dir = install_account_config(
        Path(account["kaggle_json"]),
        run_root / "runtime" / "accounts" / account["name"],
    )
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(runtime_config_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args[:2] == ["kernels", "push"] else "a"
    with log_path.open(mode, encoding="utf-8") as log:
        log.write(f"[{now_iso()}] kaggle {' '.join(args)}\n")
        log.flush()
        proc = subprocess.Popen(
            ["kaggle", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
        rc = proc.wait()
    if args[:2] == ["kernels", "push"]:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        if "Kernel push error:" in text:
            return 1
    return rc


def build_kernel_metadata(job: dict, username: str, slug: str) -> dict:
    metadata = {
        "id": f"{username}/{slug}",
        "title": slug,
        "code_file": job.get("code_file", "run_kernel.py"),
        "language": job.get("language", "python"),
        "kernel_type": job.get("kernel_type", "script"),
        "is_private": bool(job.get("is_private", True)),
        "enable_gpu": bool(job.get("enable_gpu", True)),
        "enable_internet": bool(job.get("enable_internet", False)),
        "dataset_sources": job.get("dataset_sources", DEFAULT_DATASET_SOURCES),
        "competition_sources": job.get("competition_sources", []),
        "kernel_sources": job.get("kernel_sources", []),
    }
    if job.get("machine_shape"):
        metadata["machine_shape"] = job["machine_shape"]
    return metadata


def parse_kaggle_url(log_path: Path) -> str:
    if not log_path.is_file():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"https://www\.kaggle\.com/code/\S+", text)
    return matches[-1] if matches else ""


def classify_push_failure(text: str) -> str:
    lowered = text.lower()
    if any(pattern in lowered for pattern in COMPETITION_RULES_PATTERNS):
        return "competition_rules"
    if any(pattern in text for pattern in GPU_CAPACITY_PATTERNS):
        return "gpu_capacity"
    if any(pattern in text for pattern in GPU_QUOTA_PATTERNS):
        return "gpu_quota"
    return "failed"


def requeue_after_push_backoff(run_root: Path, state_job: dict, job_id: str, reason: str) -> str:
    state_job["status"] = "queued"
    state_job["account"] = ""
    state_job["url"] = ""
    state_job["slug"] = ""
    state_job["failure_reason"] = ""
    for key in ("bundle_id", "bundle_index", "bundle_size"):
        state_job.pop(key, None)
    append_progress(run_root, f"queued job={job_id} reason={reason}")
    return reason


def stage_kernel(run_root: Path, job: dict, account: dict) -> tuple[Path, str, str]:
    job_id = job["job_id"]
    kernel_dir = run_root / "jobs" / job_id / "kernel"
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    template_dir = Path(job.get("template_dir", "kaggle/camopatch_job_template"))
    shutil.copytree(template_dir, kernel_dir)

    slug = f"{short_slug(job_id)}-{int(time.time())}"
    metadata = build_kernel_metadata(job, account["username"], slug)
    save_json(kernel_dir / "kernel-metadata.json", metadata)
    save_json(kernel_dir / "job_config.json", job.get("job_config", {}))
    code_file = kernel_dir / metadata["code_file"]
    if code_file.is_file():
        text = code_file.read_text(encoding="utf-8")
        embedded_config = repr(job.get("job_config", {}))
        text = text.replace("EMBEDDED_JOB_CONFIG = None", f"EMBEDDED_JOB_CONFIG = {embedded_config}")
        code_file.write_text(text, encoding="utf-8")
    url = f"https://www.kaggle.com/code/{account['username']}/{slug}"
    return kernel_dir, slug, url


def submit_job(run_root: Path, job: dict, state_job: dict, account: dict) -> str:
    job_id = job["job_id"]
    job_root = run_root / "jobs" / job_id
    output_dir = job_root / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_log = job_root / "output.log"
    if output_log.exists():
        output_log.unlink()
    state_job.pop("result_zip", None)
    state_job.pop("done_at", None)
    for key in ("bundle_id", "bundle_index", "bundle_size"):
        state_job.pop(key, None)
    state_job["status"] = "submitting"
    state_job["account"] = account["name"]
    state_job["submitted_at"] = now_iso()
    state_job["tries"] = int(state_job.get("tries", 0)) + 1
    kernel_dir, slug, url = stage_kernel(run_root, job, account)
    state_job["slug"] = slug
    state_job["url"] = url
    state_job["expected_zip"] = job.get("expected_zip", f"{job_id}_result.zip")
    append_progress(run_root, f"submitting job={job_id} account={account['name']}")
    push_log = run_root / "jobs" / job_id / "push.log"
    push_args = ["kernels", "push", "-p", str(kernel_dir)]
    if job.get("machine_shape"):
        push_args.extend(["--accelerator", str(job["machine_shape"])])
    rc = run_kaggle(push_args, account, run_root, push_log)
    if rc != 0:
        text = push_log.read_text(encoding="utf-8", errors="ignore") if push_log.is_file() else ""
        push_failure = classify_push_failure(text)
        if push_failure in {"gpu_capacity", "gpu_quota", "competition_rules"}:
            return requeue_after_push_backoff(run_root, state_job, job_id, push_failure)
        state_job["status"] = "failed"
        state_job["failure_reason"] = "kaggle kernels push failed"
        append_progress(run_root, f"failed job={job_id} reason=push")
        return "failed"
    actual_url = parse_kaggle_url(push_log)
    if actual_url:
        state_job["url"] = actual_url
        state_job["slug"] = actual_url.rstrip("/").rsplit("/", 1)[-1]
    state_job["status"] = "running"
    state_job["running_at"] = now_iso()
    (run_root / "jobs" / job_id / "url.txt").write_text(str(state_job["url"]) + "\n", encoding="utf-8")
    append_progress(run_root, f"running job={job_id} url={state_job['url']}")
    return "running"


def job_model(job: dict) -> str:
    return str(job.get("job_config", {}).get("model", ""))


def job_patch_size(job: dict) -> int | None:
    value = job.get("job_config", {}).get("patch_size", "")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def estimated_job_hours(job: dict) -> float:
    job_config = job.get("job_config", {})
    value = job.get("estimated_hours", job_config.get("estimated_hours"))
    if value is not None:
        try:
            return max(0.1, float(value))
        except (TypeError, ValueError):
            pass
    model = job_model(job)
    patch_size = job_patch_size(job)
    if patch_size is not None:
        estimate = DEFAULT_BUNDLE_ESTIMATE_HOURS_BY_MODEL_PATCH.get((model, patch_size))
        if estimate is not None:
            return estimate
    return DEFAULT_BUNDLE_ESTIMATE_HOURS.get(model, 3.0)


def account_float(account: dict, key: str, default: float) -> float:
    try:
        return float(account.get(key, default))
    except (TypeError, ValueError):
        return default


def account_int(account: dict, key: str, default: int) -> int:
    try:
        return int(account.get(key, default))
    except (TypeError, ValueError):
        return default


def quota_reset_window(
    now: datetime,
    reset_weekday: int,
    reset_hour: int,
    reset_timezone: str,
) -> tuple[datetime, datetime, str]:
    try:
        tz = ZoneInfo(reset_timezone)
        resolved_timezone = reset_timezone
    except ZoneInfoNotFoundError:
        tz = timezone.utc
        resolved_timezone = "UTC"
    reset_weekday = max(0, min(6, int(reset_weekday)))
    reset_hour = max(0, min(23, int(reset_hour)))
    local_now = now.astimezone(tz)
    reset_today = local_now.replace(hour=reset_hour, minute=0, second=0, microsecond=0)
    days_since_reset_day = (local_now.weekday() - reset_weekday) % 7
    window_start = reset_today - timedelta(days=days_since_reset_day)
    if window_start > local_now:
        window_start -= timedelta(days=7)
    next_reset = window_start + timedelta(days=7)
    return window_start.astimezone(timezone.utc), next_reset.astimezone(timezone.utc), resolved_timezone


def job_quota_unit(job_id: str, state_job: dict) -> str:
    return str(state_job.get("bundle_id") or state_job.get("slug") or job_id)


def estimate_account_quota(
    state: dict,
    account: dict,
    args: argparse.Namespace,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    weekly_quota_hours = account_float(account, "weekly_gpu_quota_hours", float(args.weekly_gpu_quota_hours))
    reset_weekday = account_int(account, "quota_reset_weekday", int(args.quota_reset_weekday))
    reset_hour = account_int(account, "quota_reset_hour", int(args.quota_reset_hour))
    reset_timezone = str(account.get("quota_reset_timezone", args.quota_reset_timezone))
    window_start, next_reset, resolved_timezone = quota_reset_window(now, reset_weekday, reset_hour, reset_timezone)

    units: dict[str, dict[str, Any]] = {}
    for job_id, state_job in state.get("jobs", {}).items():
        if state_job.get("account") != account["name"]:
            continue
        start = parse_iso(str(state_job.get("running_at") or state_job.get("submitted_at") or ""))
        if start is None:
            continue
        status = str(state_job.get("status", ""))
        active = status in ACTIVE_STATUSES
        end = now if active else parse_iso(
            str(state_job.get("done_at") or state_job.get("last_checked") or state_job.get("submitted_at") or "")
        )
        if end is None:
            end = now
        if end <= window_start:
            continue
        unit_key = job_quota_unit(job_id, state_job)
        unit = units.setdefault(
            unit_key,
            {
                "start": start,
                "end": end,
                "active": active,
                "jobs": 0,
            },
        )
        unit["start"] = min(unit["start"], start)
        unit["end"] = max(unit["end"], end)
        unit["active"] = bool(unit["active"] or active)
        unit["jobs"] = int(unit["jobs"]) + 1

    used_hours = 0.0
    active_hours = 0.0
    active_units = 0
    for unit in units.values():
        start = max(unit["start"], window_start)
        end = now if unit["active"] else min(unit["end"], now)
        if end <= start:
            continue
        elapsed_hours = (end - start).total_seconds() / 3600.0
        used_hours += elapsed_hours
        if unit["active"]:
            active_hours += elapsed_hours
            active_units += 1

    remaining_hours = weekly_quota_hours - used_hours
    under_hours = account_float(
        account,
        "auto_bundle_under_quota_hours",
        float(args.auto_bundle_under_quota_hours),
    )
    auto_bundle = bool(under_hours > 0 and remaining_hours <= under_hours)
    return {
        "account": account["name"],
        "username": account["username"],
        "weekly_quota_hours": round(weekly_quota_hours, 3),
        "used_hours": round(used_hours, 3),
        "active_hours": round(active_hours, 3),
        "remaining_hours": round(remaining_hours, 3),
        "active_units": active_units,
        "observed_units": len(units),
        "window_start": window_start.replace(microsecond=0).isoformat(),
        "next_reset": next_reset.replace(microsecond=0).isoformat(),
        "reset_weekday": reset_weekday,
        "reset_hour": reset_hour,
        "reset_timezone": resolved_timezone,
        "auto_bundle_under_quota_hours": round(under_hours, 3),
        "auto_bundle": auto_bundle,
        "generated_at": now.replace(microsecond=0).isoformat(),
    }


def estimate_account_quotas(state: dict, accounts: list[dict], args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    now = datetime.now(timezone.utc)
    return {account["name"]: estimate_account_quota(state, account, args, now) for account in accounts}


def bundle_target_for_account(account: dict, args: argparse.Namespace, quota_estimate: dict[str, Any]) -> float:
    manual_target = float(args.bundle_target_hours)
    if manual_target > 0:
        return manual_target
    if quota_estimate.get("auto_bundle"):
        return account_float(account, "auto_bundle_target_hours", float(args.auto_bundle_target_hours))
    return 0.0


def bundle_max_jobs_for_account(account: dict, args: argparse.Namespace) -> int:
    return max(1, account_int(account, "bundle_max_jobs", int(args.bundle_max_jobs)))


def select_bundle_job_ids(
    state: dict,
    jobs_by_id: dict[str, dict],
    first_id: str,
    target_hours: float,
    max_jobs: int,
) -> list[str]:
    if target_hours <= 0 or max_jobs <= 1:
        return [first_id]
    first_job = jobs_by_id[first_id]
    model = job_model(first_job)
    selected: list[str] = []
    total_hours = 0.0
    for job_id, state_job in state.get("jobs", {}).items():
        if state_job.get("status") != "queued" or job_id not in jobs_by_id:
            continue
        job = jobs_by_id[job_id]
        if job_model(job) != model:
            continue
        estimate = estimated_job_hours(job)
        if selected and (total_hours + estimate) > target_hours:
            break
        selected.append(job_id)
        total_hours += estimate
        if len(selected) >= max_jobs:
            break
    return selected or [first_id]


def bundle_job_id(job_ids: list[str]) -> str:
    first = sanitize_slug(job_ids[0].removeprefix("camopatch-bcos-"))
    return f"bundle-{first[:30]}-n{len(job_ids)}-{int(time.time())}"


def make_bundle_job(bundle_id: str, bundle_jobs: list[dict]) -> dict:
    first = bundle_jobs[0]
    first_config = first.get("job_config", {})
    estimated_hours = sum(estimated_job_hours(job) for job in bundle_jobs)
    return {
        "job_id": bundle_id,
        "title": f"CamoPatch bundle {len(bundle_jobs)} jobs",
        "template_dir": first.get("template_dir", "kaggle/camopatch_job_template"),
        "code_file": first.get("code_file", "run_kernel.py"),
        "kernel_type": first.get("kernel_type", "script"),
        "language": first.get("language", "python"),
        "enable_gpu": bool(first.get("enable_gpu", True)),
        "enable_internet": bool(first.get("enable_internet", False)),
        "machine_shape": first.get("machine_shape"),
        "is_private": bool(first.get("is_private", True)),
        "dataset_sources": first.get("dataset_sources", DEFAULT_DATASET_SOURCES),
        "competition_sources": first.get("competition_sources", []),
        "kernel_sources": first.get("kernel_sources", []),
        "output_pattern": ".*(zip|log|json|csv)$",
        "expected_zip": f"{bundle_id}_bundle_result.zip",
        "timeout_minutes": max(DEFAULT_BUNDLE_TIMEOUT_MINUTES, int((estimated_hours + 3.0) * 60)),
        "job_config": {
            "job_id": bundle_id,
            "bundle": True,
            "bundle_size": len(bundle_jobs),
            "estimated_hours": round(estimated_hours, 3),
            "code_dataset_owner": first_config.get("code_dataset_owner", "hkhnhduy"),
            "code_dataset_slug": first_config.get("code_dataset_slug", "attack-bcos-github"),
            "bundle_jobs": [dict(job.get("job_config", {})) for job in bundle_jobs],
        },
    }


def submit_bundle(
    run_root: Path,
    jobs_by_id: dict[str, dict],
    state: dict,
    job_ids: list[str],
    account: dict,
) -> str:
    if len(job_ids) == 1:
        job_id = job_ids[0]
        return submit_job(run_root, jobs_by_id[job_id], state["jobs"][job_id], account)

    bundle_id = bundle_job_id(job_ids)
    bundle_jobs = [jobs_by_id[job_id] for job_id in job_ids]
    bundle = make_bundle_job(bundle_id, bundle_jobs)
    bundle_root = run_root / "bundles" / bundle_id
    output_dir = bundle_root / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    bundle_root.mkdir(parents=True, exist_ok=True)

    submitted_at = now_iso()
    for idx, job_id in enumerate(job_ids, start=1):
        state_job = state["jobs"][job_id]
        state_job.pop("result_zip", None)
        state_job.pop("done_at", None)
        state_job["status"] = "submitting"
        state_job["account"] = account["name"]
        state_job["submitted_at"] = submitted_at
        state_job["tries"] = int(state_job.get("tries", 0)) + 1
        state_job["bundle_id"] = bundle_id
        state_job["bundle_index"] = idx
        state_job["bundle_size"] = len(job_ids)
        state_job["bundle_timeout_minutes"] = bundle["timeout_minutes"]
        state_job["expected_zip"] = jobs_by_id[job_id].get("expected_zip", f"{job_id}_result.zip")
        state_job["failure_reason"] = ""

    kernel_dir, slug, url = stage_kernel(run_root, bundle, account)
    for job_id in job_ids:
        state["jobs"][job_id]["slug"] = slug
        state["jobs"][job_id]["url"] = url
    append_progress(
        run_root,
        f"submitting_bundle bundle={bundle_id} account={account['name']} jobs={','.join(job_ids)}",
    )
    push_log = run_root / "jobs" / bundle_id / "push.log"
    push_args = ["kernels", "push", "-p", str(kernel_dir)]
    if bundle.get("machine_shape"):
        push_args.extend(["--accelerator", str(bundle["machine_shape"])])
    rc = run_kaggle(push_args, account, run_root, push_log)
    if rc != 0:
        text = push_log.read_text(encoding="utf-8", errors="ignore") if push_log.is_file() else ""
        push_failure = classify_push_failure(text)
        if push_failure in {"gpu_capacity", "gpu_quota", "competition_rules"}:
            for job_id in job_ids:
                requeue_after_push_backoff(run_root, state["jobs"][job_id], job_id, push_failure)
            return push_failure
        for job_id in job_ids:
            state_job = state["jobs"][job_id]
            state_job["status"] = "failed"
            state_job["failure_reason"] = "kaggle bundle push failed"
        append_progress(run_root, f"failed_bundle bundle={bundle_id} reason=push")
        return "failed"

    actual_url = parse_kaggle_url(push_log)
    if actual_url:
        url = actual_url
        slug = actual_url.rstrip("/").rsplit("/", 1)[-1]
    running_at = now_iso()
    for job_id in job_ids:
        state_job = state["jobs"][job_id]
        state_job["url"] = url
        state_job["slug"] = slug
        state_job["status"] = "running"
        state_job["running_at"] = running_at
        job_root = run_root / "jobs" / job_id
        job_root.mkdir(parents=True, exist_ok=True)
        (job_root / "url.txt").write_text(url + "\n", encoding="utf-8")
    (bundle_root / "url.txt").write_text(url + "\n", encoding="utf-8")
    append_progress(run_root, f"running_bundle bundle={bundle_id} url={url}")
    return "running"


def account_active_count(state: dict, account_name: str) -> int:
    active_units: set[str] = set()
    for job_id, job in state.get("jobs", {}).items():
        if job.get("account") != account_name or job.get("status") not in ACTIVE_STATUSES:
            continue
        active_units.add(str(job.get("bundle_id") or job.get("slug") or job_id))
    return len(active_units)


def find_expected_zip(output_dir: Path, expected_zip: str) -> Path | None:
    direct = output_dir / expected_zip
    if direct.is_file():
        return direct
    matches = sorted(output_dir.rglob(expected_zip))
    return matches[0] if matches else None


def count_csv_rows(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def expected_summary_rows(job: dict) -> int | None:
    job_config = job.get("job_config", {})
    images_csv = job_config.get("images_csv")
    if not images_csv:
        return None
    rows = count_csv_rows(Path(images_csv))
    limit_images = int(job_config.get("limit_images", 0) or 0)
    if rows is not None and limit_images > 0:
        return min(rows, limit_images)
    return rows


def validate_result_zip(path: Path, expected_rows: int | None) -> tuple[bool, str]:
    try:
        with zipfile.ZipFile(path) as archive:
            bad_file = archive.testzip()
            if bad_file is not None:
                return False, f"bad zip entry: {bad_file}"
            names = set(archive.namelist())
            if "manifest.json" not in names:
                return False, "missing manifest.json"
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            if int(manifest.get("return_code", -1)) != 0:
                return False, f"manifest return_code={manifest.get('return_code')}"
            if "outputs/summary.csv" not in names:
                return False, "missing outputs/summary.csv"
            if "outputs/success_events.csv" not in names:
                return False, "missing outputs/success_events.csv"
            if "outputs/success_by_query.csv" not in names:
                return False, "missing outputs/success_by_query.csv"
            if expected_rows is not None:
                summary_text = archive.read("outputs/summary.csv").decode("utf-8")
                reader = csv.DictReader(io.StringIO(summary_text))
                if "first_success_query" not in (reader.fieldnames or []):
                    return False, "summary missing first_success_query"
                actual_rows = sum(1 for _ in reader)
                if actual_rows != expected_rows:
                    return False, f"summary rows {actual_rows} != expected {expected_rows}"
            return True, "ok"
    except zipfile.BadZipFile:
        return False, "bad zip file"
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        return False, f"invalid result metadata: {exc}"


def log_has_fatal(output_dir: Path) -> bool:
    fatal_patterns = ("traceback (most recent call last)", "runtimeerror:", "valueerror:", "cuda out of memory")
    for path in output_dir.rglob("*.log"):
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if any(pattern in text for pattern in fatal_patterns):
            return True
    return False


def minutes_since(value: str) -> float:
    dt = parse_iso(value)
    if dt is None:
        return 0.0
    return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0


def iso_after_minutes(minutes: float) -> str:
    return (
        datetime.fromtimestamp(time.time() + minutes * 60, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def is_future_iso(value: str) -> bool:
    dt = parse_iso(value)
    if dt is None:
        return False
    return dt > datetime.now(timezone.utc)


def poll_job(run_root: Path, job: dict, state_job: dict, accounts_by_name: dict[str, dict]) -> None:
    account = accounts_by_name.get(state_job.get("account", ""))
    if account is None:
        state_job["status"] = "failed"
        state_job["failure_reason"] = "missing account for running job"
        return

    job_id = job["job_id"]
    state_job["status"] = "downloading"
    output_dir = run_root / "jobs" / job_id / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    kernel_ref = f"{account['username']}/{state_job['slug']}"
    pattern = job.get("output_pattern", ".*(zip|log)$")
    rc = run_kaggle(
        ["kernels", "output", kernel_ref, "-p", str(output_dir), "--file-pattern", pattern, "-o"],
        account,
        run_root,
        run_root / "jobs" / job_id / "output.log",
    )
    state_job["last_checked"] = now_iso()

    expected_zip = state_job.get("expected_zip", f"{job_id}_result.zip")
    zip_path = find_expected_zip(output_dir, expected_zip)
    expected_rows = expected_summary_rows(job)
    if zip_path:
        is_valid_zip, validation_reason = validate_result_zip(zip_path, expected_rows)
    else:
        is_valid_zip, validation_reason = False, ""
    if zip_path and is_valid_zip:
        state_job["status"] = "done"
        state_job["done_at"] = now_iso()
        state_job["result_zip"] = str(zip_path)
        append_progress(run_root, f"done job={job_id}")
        return
    if zip_path and not is_valid_zip:
        state_job["status"] = "failed"
        state_job["failure_reason"] = f"invalid result zip: {validation_reason}"
        append_progress(run_root, f"failed job={job_id} reason=invalid_zip detail={validation_reason}")
        return
    if log_has_fatal(output_dir):
        state_job["status"] = "failed"
        state_job["failure_reason"] = "fatal log pattern"
        append_progress(run_root, f"failed job={job_id} reason=fatal_log")
        return
    if minutes_since(state_job.get("submitted_at", "")) > float(job.get("timeout_minutes", 720)):
        state_job["status"] = "failed"
        state_job["failure_reason"] = "timeout"
        append_progress(run_root, f"failed job={job_id} reason=timeout")
        return

    state_job["status"] = "running"
    if rc != 0:
        append_progress(run_root, f"poll_pending job={job_id}")


def poll_bundle(
    run_root: Path,
    bundle_id: str,
    state: dict,
    jobs_by_id: dict[str, dict],
    accounts_by_name: dict[str, dict],
) -> None:
    active_items = [
        (job_id, state_job)
        for job_id, state_job in state.get("jobs", {}).items()
        if state_job.get("bundle_id") == bundle_id
        and state_job.get("status") in ACTIVE_STATUSES
        and job_id in jobs_by_id
    ]
    if not active_items:
        return

    account = accounts_by_name.get(str(active_items[0][1].get("account", "")))
    if account is None:
        for job_id, state_job in active_items:
            state_job["status"] = "failed"
            state_job["failure_reason"] = "missing account for running bundle"
            append_progress(run_root, f"failed job={job_id} reason=missing_bundle_account")
        return

    for _, state_job in active_items:
        state_job["status"] = "downloading"
    output_dir = run_root / "bundles" / bundle_id / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = str(active_items[0][1].get("slug", ""))
    kernel_ref = f"{account['username']}/{slug}"
    rc = run_kaggle(
        ["kernels", "output", kernel_ref, "-p", str(output_dir), "--file-pattern", ".*(zip|log|json|csv)$", "-o"],
        account,
        run_root,
        run_root / "bundles" / bundle_id / "output.log",
    )
    checked_at = now_iso()
    has_fatal = log_has_fatal(output_dir)

    for job_id, state_job in active_items:
        state_job["last_checked"] = checked_at
        expected_zip = state_job.get("expected_zip", f"{job_id}_result.zip")
        zip_path = find_expected_zip(output_dir, expected_zip)
        expected_rows = expected_summary_rows(jobs_by_id[job_id])
        if zip_path:
            is_valid_zip, validation_reason = validate_result_zip(zip_path, expected_rows)
        else:
            is_valid_zip, validation_reason = False, ""
        if zip_path and is_valid_zip:
            state_job["status"] = "done"
            state_job["done_at"] = now_iso()
            state_job["result_zip"] = str(zip_path)
            append_progress(run_root, f"done job={job_id} bundle={bundle_id}")
            continue
        if zip_path and not is_valid_zip:
            state_job["status"] = "failed"
            state_job["failure_reason"] = f"invalid result zip: {validation_reason}"
            append_progress(
                run_root,
                f"failed job={job_id} bundle={bundle_id} reason=invalid_zip detail={validation_reason}",
            )
            continue
        if has_fatal and rc == 0:
            state_job["status"] = "failed"
            state_job["failure_reason"] = "fatal bundle log pattern"
            append_progress(run_root, f"failed job={job_id} bundle={bundle_id} reason=fatal_log")
            continue
        timeout_minutes = max(
            float(jobs_by_id[job_id].get("timeout_minutes", 720)),
            float(state_job.get("bundle_timeout_minutes", DEFAULT_BUNDLE_TIMEOUT_MINUTES) or DEFAULT_BUNDLE_TIMEOUT_MINUTES),
        )
        if minutes_since(state_job.get("submitted_at", "")) > timeout_minutes:
            state_job["status"] = "failed"
            state_job["failure_reason"] = "timeout"
            append_progress(run_root, f"failed job={job_id} bundle={bundle_id} reason=timeout")
            continue
        state_job["status"] = "running"

    if rc != 0:
        append_progress(run_root, f"poll_pending_bundle bundle={bundle_id}")


def scheduler_loop(args: argparse.Namespace) -> None:
    run_root = args.run_root
    run_root.mkdir(parents=True, exist_ok=True)
    accounts = discover_accounts(run_root, args.accounts_config)
    accounts_by_name = {account["name"]: account for account in accounts}
    state_path = run_root / "state.json"
    state = load_json(state_path, {"jobs": {}})
    state.setdefault("account_backoff_until", {})
    state.setdefault("account_quota_estimates", {})

    while True:
        jobs_by_id = load_jobs(args.jobs_config)
        ensure_jobs_in_state(state, jobs_by_id, run_root)
        state.setdefault("account_backoff_until", {})
        state.setdefault("account_quota_estimates", {})

        polled_bundles: set[str] = set()
        for job_id, state_job in list(state.get("jobs", {}).items()):
            if state_job.get("status") in ACTIVE_STATUSES and job_id in jobs_by_id:
                try:
                    bundle_id = str(state_job.get("bundle_id", ""))
                    if bundle_id:
                        if bundle_id in polled_bundles:
                            continue
                        polled_bundles.add(bundle_id)
                        poll_bundle(run_root, bundle_id, state, jobs_by_id, accounts_by_name)
                    else:
                        poll_job(run_root, jobs_by_id[job_id], state_job, accounts_by_name)
                except Exception as exc:  # pragma: no cover - defensive long-run guard
                    state_job["status"] = "running"
                    state_job["failure_reason"] = f"poll exception: {exc}"
                    append_progress(run_root, f"poll_exception job={job_id} detail={exc}")
                save_json(state_path, state)
                write_dashboard(run_root, state)

        state["account_quota_estimates"] = estimate_account_quotas(state, accounts, args)
        save_json(state_path, state)
        write_dashboard(run_root, state)

        submitted = 0
        if not args.poll_only:
            for account in accounts:
                backoff_until = str(state["account_backoff_until"].get(account["name"], ""))
                if is_future_iso(backoff_until):
                    append_progress(
                        run_root,
                        f"skip_account account={account['name']} reason=gpu_capacity_until {backoff_until}",
                    )
                    continue
                quota_estimate = state.get("account_quota_estimates", {}).get(account["name"], {})
                bundle_target_hours = bundle_target_for_account(account, args, quota_estimate)
                bundle_max_jobs = bundle_max_jobs_for_account(account, args)
                free_slots = int(account["max_running"]) - account_active_count(state, account["name"])
                while free_slots > 0:
                    next_id = next(
                        (
                            job_id
                            for job_id, state_job in state["jobs"].items()
                            if state_job.get("status") == "queued" and job_id in jobs_by_id
                        ),
                        None,
                    )
                    if next_id is None:
                        break
                    bundle_ids = [next_id]
                    try:
                        bundle_ids = select_bundle_job_ids(
                            state,
                            jobs_by_id,
                            next_id,
                            bundle_target_hours,
                            bundle_max_jobs,
                        )
                        if bundle_target_hours > 0 and len(bundle_ids) > 1:
                            bundle_reason = (
                                "quota"
                                if quota_estimate.get("auto_bundle") and float(args.bundle_target_hours) <= 0
                                else "manual"
                            )
                            append_progress(
                                run_root,
                                "bundle_select "
                                f"reason={bundle_reason} "
                                f"account={account['name']} "
                                f"remaining_hours={quota_estimate.get('remaining_hours', 'unknown')} "
                                f"target_hours={bundle_target_hours} "
                                f"jobs={len(bundle_ids)}",
                            )
                        result = submit_bundle(run_root, jobs_by_id, state, bundle_ids, account)
                    except Exception as exc:  # pragma: no cover - defensive long-run guard
                        for failed_id in bundle_ids:
                            state["jobs"][failed_id]["status"] = "queued"
                            state["jobs"][failed_id]["account"] = ""
                            state["jobs"][failed_id]["url"] = ""
                            state["jobs"][failed_id]["slug"] = ""
                            state["jobs"][failed_id]["failure_reason"] = f"submit exception: {exc}"
                            for key in ("bundle_id", "bundle_index", "bundle_size"):
                                state["jobs"][failed_id].pop(key, None)
                        result = "exception"
                        append_progress(
                            run_root,
                            f"submit_exception jobs={','.join(bundle_ids)} account={account['name']} detail={exc}",
                        )
                    if result in {"gpu_capacity", "gpu_quota", "competition_rules"}:
                        if result == "gpu_quota":
                            cooldown_minutes = float(args.quota_cooldown_minutes)
                        elif result == "competition_rules":
                            cooldown_minutes = float(args.rules_cooldown_minutes)
                        else:
                            cooldown_minutes = float(args.account_cooldown_minutes)
                        backoff_until = iso_after_minutes(cooldown_minutes)
                        state["account_backoff_until"][account["name"]] = backoff_until
                        append_progress(
                            run_root,
                            f"cooldown account={account['name']} reason={result} until={backoff_until}",
                        )
                        break
                    submitted += 1
                    free_slots -= 1
                    save_json(state_path, state)
                    write_dashboard(run_root, state)
                    if args.max_submit and submitted >= args.max_submit:
                        break
                if args.max_submit and submitted >= args.max_submit:
                    break

        state["account_quota_estimates"] = estimate_account_quotas(state, accounts, args)
        save_json(state_path, state)
        write_dashboard(run_root, state)

        statuses = {job.get("status") for job in state.get("jobs", {}).values()}
        if args.once or (args.exit_when_done and statuses <= TERMINAL_STATUSES):
            break
        time.sleep(args.poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit and monitor CamoPatch Kaggle jobs.")
    parser.add_argument("--jobs-config", type=Path, default=Path("kaggle/camopatch_jobs.json"))
    parser.add_argument("--accounts-config", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=Path("kaggle_runs"))
    parser.add_argument("--poll-interval", type=int, default=300)
    parser.add_argument("--max-submit", type=int, default=0, help="Maximum jobs to submit in this invocation. 0 = no cap.")
    parser.add_argument(
        "--account-cooldown-minutes",
        type=float,
        default=30.0,
        help="Cooldown an account this long after Kaggle reports no free GPU sessions.",
    )
    parser.add_argument(
        "--quota-cooldown-minutes",
        type=float,
        default=1440.0,
        help="Cooldown an account this long after Kaggle reports weekly GPU quota exhaustion.",
    )
    parser.add_argument(
        "--rules-cooldown-minutes",
        type=float,
        default=10080.0,
        help="Cooldown an account this long after Kaggle reports unaccepted competition rules.",
    )
    parser.add_argument("--poll-only", action="store_true")
    parser.add_argument("--once", action="store_true", help="Run one poll/submit cycle and exit.")
    parser.add_argument("--exit-when-done", action="store_true")
    parser.add_argument(
        "--bundle-target-hours",
        type=float,
        default=0.0,
        help="If >0, submit queued jobs from the same model as sequential bundles near this target runtime.",
    )
    parser.add_argument(
        "--bundle-max-jobs",
        type=int,
        default=5,
        help="Maximum subjobs per bundle when --bundle-target-hours is enabled.",
    )
    parser.add_argument(
        "--weekly-gpu-quota-hours",
        type=float,
        default=DEFAULT_WEEKLY_GPU_QUOTA_HOURS,
        help="Estimated weekly GPU quota per account; account config can override weekly_gpu_quota_hours.",
    )
    parser.add_argument(
        "--quota-reset-weekday",
        type=int,
        default=DEFAULT_QUOTA_RESET_WEEKDAY,
        help="Estimated quota reset weekday using Python numbering, Monday=0 ... Saturday=5.",
    )
    parser.add_argument(
        "--quota-reset-hour",
        type=int,
        default=DEFAULT_QUOTA_RESET_HOUR,
        help="Estimated quota reset hour in --quota-reset-timezone.",
    )
    parser.add_argument(
        "--quota-reset-timezone",
        default=DEFAULT_QUOTA_RESET_TIMEZONE,
        help="Timezone for estimated quota reset, for example UTC or Europe/Berlin.",
    )
    parser.add_argument(
        "--auto-bundle-under-quota-hours",
        type=float,
        default=DEFAULT_AUTO_BUNDLE_UNDER_QUOTA_HOURS,
        help="Automatically bundle when estimated remaining weekly quota is at or below this many hours. 0 disables.",
    )
    parser.add_argument(
        "--auto-bundle-target-hours",
        type=float,
        default=DEFAULT_AUTO_BUNDLE_TARGET_HOURS,
        help="Target runtime for automatic quota-tail bundles.",
    )
    args = parser.parse_args()
    scheduler_loop(args)


if __name__ == "__main__":
    main()
