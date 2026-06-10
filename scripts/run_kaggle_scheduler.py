#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ACTIVE_STATUSES = {"submitting", "running", "downloading"}
TERMINAL_STATUSES = {"done", "failed"}
DEFAULT_DATASET_SOURCES = [
    "hkhnhduy/attack-bcos-github",
    "hkhnhduy/weights-bcos",
    "sautkin/imagenet1kvalid",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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
    return slug[:45].strip("-")


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
        normalized.append(
            {
                "name": name,
                "kaggle_json": str(kaggle_json),
                "max_running": int(account.get("max_running", 2)),
                "username": read_kaggle_username(kaggle_json),
            }
        )
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
    with log_path.open("a", encoding="utf-8") as log:
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
        return proc.wait()


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
        embedded_config = json.dumps(job.get("job_config", {}), indent=2, sort_keys=True)
        text = text.replace("EMBEDDED_JOB_CONFIG = None", f"EMBEDDED_JOB_CONFIG = {embedded_config}")
        code_file.write_text(text, encoding="utf-8")
    url = f"https://www.kaggle.com/code/{account['username']}/{slug}"
    return kernel_dir, slug, url


def submit_job(run_root: Path, job: dict, state_job: dict, account: dict) -> None:
    job_id = job["job_id"]
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
    rc = run_kaggle(["kernels", "push", "-p", str(kernel_dir)], account, run_root, push_log)
    if rc != 0:
        state_job["status"] = "failed"
        state_job["failure_reason"] = "kaggle kernels push failed"
        append_progress(run_root, f"failed job={job_id} reason=push")
        return
    actual_url = parse_kaggle_url(push_log)
    if actual_url:
        state_job["url"] = actual_url
        state_job["slug"] = actual_url.rstrip("/").rsplit("/", 1)[-1]
    state_job["status"] = "running"
    state_job["running_at"] = now_iso()
    (run_root / "jobs" / job_id / "url.txt").write_text(str(state_job["url"]) + "\n", encoding="utf-8")
    append_progress(run_root, f"running job={job_id} url={state_job['url']}")


def account_active_count(state: dict, account_name: str) -> int:
    return sum(
        1
        for job in state.get("jobs", {}).values()
        if job.get("account") == account_name and job.get("status") in ACTIVE_STATUSES
    )


def find_expected_zip(output_dir: Path, expected_zip: str) -> Path | None:
    direct = output_dir / expected_zip
    if direct.is_file():
        return direct
    matches = sorted(output_dir.rglob(expected_zip))
    return matches[0] if matches else None


def valid_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except zipfile.BadZipFile:
        return False


def log_has_fatal(output_dir: Path) -> bool:
    fatal_patterns = ("traceback (most recent call last)", "runtimeerror:", "valueerror:", "cuda out of memory")
    for path in output_dir.rglob("*.log"):
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if any(pattern in text for pattern in fatal_patterns):
            return True
    return False


def minutes_since(value: str) -> float:
    if not value:
        return 0.0
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0


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
    if zip_path and valid_zip(zip_path):
        state_job["status"] = "done"
        state_job["done_at"] = now_iso()
        state_job["result_zip"] = str(zip_path)
        append_progress(run_root, f"done job={job_id}")
        return
    if zip_path and not valid_zip(zip_path):
        state_job["status"] = "failed"
        state_job["failure_reason"] = "invalid result zip"
        append_progress(run_root, f"failed job={job_id} reason=invalid_zip")
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


def scheduler_loop(args: argparse.Namespace) -> None:
    run_root = args.run_root
    run_root.mkdir(parents=True, exist_ok=True)
    accounts = discover_accounts(run_root, args.accounts_config)
    accounts_by_name = {account["name"]: account for account in accounts}
    state_path = run_root / "state.json"
    state = load_json(state_path, {"jobs": {}})

    while True:
        jobs_by_id = load_jobs(args.jobs_config)
        ensure_jobs_in_state(state, jobs_by_id, run_root)

        for job_id, state_job in list(state.get("jobs", {}).items()):
            if state_job.get("status") in ACTIVE_STATUSES and job_id in jobs_by_id:
                poll_job(run_root, jobs_by_id[job_id], state_job, accounts_by_name)
                save_json(state_path, state)
                write_dashboard(run_root, state)

        submitted = 0
        if not args.poll_only:
            for account in accounts:
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
                    submit_job(run_root, jobs_by_id[next_id], state["jobs"][next_id], account)
                    submitted += 1
                    free_slots -= 1
                    save_json(state_path, state)
                    write_dashboard(run_root, state)
                    if args.max_submit and submitted >= args.max_submit:
                        break
                if args.max_submit and submitted >= args.max_submit:
                    break

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
    parser.add_argument("--poll-only", action="store_true")
    parser.add_argument("--once", action="store_true", help="Run one poll/submit cycle and exit.")
    parser.add_argument("--exit-when-done", action="store_true")
    args = parser.parse_args()
    scheduler_loop(args)


if __name__ == "__main__":
    main()
