from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_CONFIG = "job_config.json"
DEFAULT_CODE_DATASET = "attack-bcos-github"
DEFAULT_CODE_OWNER = "hkhnhduy"
DEFAULT_RESULT_ROOT = Path("/kaggle/working/result")
DEFAULT_WORK_REPO = Path("/kaggle/working/attack_bcos")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if "bundle_jobs" in config:
        if not config.get("job_id"):
            raise ValueError(f"{path} bundle config is missing required key: job_id")
        bundle_jobs = config.get("bundle_jobs")
        if not isinstance(bundle_jobs, list) or not bundle_jobs:
            raise ValueError(f"{path} bundle config must contain a non-empty bundle_jobs list.")
        for idx, job in enumerate(bundle_jobs, start=1):
            if not isinstance(job, dict):
                raise ValueError(f"{path} bundle_jobs[{idx}] must be an object.")
            missing = [
                name
                for name in ("job_id", "model", "patch_size", "linf", "position", "queries")
                if name not in job
            ]
            if missing:
                raise ValueError(
                    f"{path} bundle_jobs[{idx}] is missing required keys: {', '.join(missing)}"
                )
        return config
    required = ["job_id", "model", "patch_size", "linf", "position", "queries"]
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"{path} is missing required keys: {', '.join(missing)}")
    return config


def candidate_code_roots(config: dict) -> Iterable[Path]:
    dataset_slug = config.get("code_dataset_slug", DEFAULT_CODE_DATASET)
    owner = config.get("code_dataset_owner", DEFAULT_CODE_OWNER)
    yield Path.cwd()
    yield Path("/kaggle/working")
    yield Path("/kaggle/input") / dataset_slug
    yield Path("/kaggle/input/datasets") / owner / dataset_slug
    yield Path("/kaggle/input/datasets") / dataset_slug
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        yield from sorted(kaggle_input.rglob(dataset_slug))
        yield from sorted(path.parent for path in kaggle_input.rglob("CamoPatch/ConCamoPatchBatch.py"))


def find_code_root(config: dict) -> Path:
    for root in candidate_code_roots(config):
        if (root / "CamoPatch" / "ConCamoPatchBatch.py").is_file():
            return root.resolve()
    raise FileNotFoundError("Could not find attack-bcos code dataset under /kaggle/input.")


def prepare_work_repo(config: dict) -> Path:
    code_root = find_code_root(config)
    work_repo = Path(config.get("work_repo", DEFAULT_WORK_REPO))
    if code_root.resolve() == work_repo.resolve():
        return work_repo
    if work_repo.exists():
        shutil.rmtree(work_repo)
    ignore = shutil.ignore_patterns(
        ".git",
        "__pycache__",
        "*.pyc",
        "artifacts",
        "result",
        "results",
        "outputs",
        "kaggle_runs",
    )
    shutil.copytree(code_root, work_repo, ignore=ignore)
    return work_repo


def find_imagenet_root() -> Path:
    candidates = [
        Path("/kaggle/input/imagenet1kvalid"),
        Path("/kaggle/input/datasets/sautkin/imagenet1kvalid"),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for candidate in sorted(kaggle_input.rglob("imagenet1kvalid")):
            if candidate.is_dir():
                return candidate
    raise FileNotFoundError("Could not find ImageNet root. Attach sautkin/imagenet1kvalid.")


def rewrite_image_path(raw_path: str, imagenet_root: Path) -> str:
    path = Path(raw_path)
    if path.is_file():
        return str(path)

    marker = "imagenet1kvalid/"
    normalized = raw_path.replace("\\", "/")
    if marker in normalized:
        rel = normalized.split(marker, 1)[1]
        return str(imagenet_root / rel)

    if not path.is_absolute():
        return str(imagenet_root / normalized)

    return raw_path


def prepare_images_csv(repo: Path, config: dict, run_dir: Path) -> Path:
    csv_path = Path(config.get("images_csv", "data/used_images_1000.csv"))
    if not csv_path.is_absolute():
        csv_path = repo / csv_path
    if not csv_path.is_file():
        raise FileNotFoundError(f"Images CSV not found: {csv_path}")

    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if "image_path" not in fieldnames:
        raise ValueError(f"{csv_path} must contain an image_path column.")
    if not rows:
        raise ValueError(f"{csv_path} has no image rows.")

    first_path = rows[0].get("image_path", "")
    if first_path and Path(first_path).is_file():
        return csv_path

    imagenet_root = find_imagenet_root()
    runtime_csv = run_dir / "images_runtime.csv"
    with runtime_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["image_path"] = rewrite_image_path(row.get("image_path", ""), imagenet_root)
            writer.writerow(row)
    return runtime_csv


def find_bcos_weights_dir() -> Path | None:
    for env_name in ("BCOS_WEIGHTS_DIR", "MODEL_WEIGHTS_DIR", "WEIGHTS_DIR"):
        env_value = os.environ.get(env_name)
        if env_value and Path(env_value).is_dir():
            return Path(env_value)

    candidates = [
        Path("/kaggle/input/weights-bcos/bcos-imagenet"),
        Path("/kaggle/input/weights-bcos"),
        Path("/kaggle/input/datasets/hkhnhduy/weights-bcos/bcos-imagenet"),
        Path("/kaggle/input/datasets/hkhnhduy/weights-bcos"),
        Path("/kaggle/working/weights/bcos-imagenet"),
        Path("/kaggle/working/weights"),
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.rglob("*.pth")):
            return candidate

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for candidate in sorted(kaggle_input.rglob("*.pth")):
            if "bcos" in str(candidate).lower() or candidate.name.startswith(("resnet_", "densenet_", "convnext_", "bcos_")):
                return candidate.parent
    return None


def position_rule_for(config: dict) -> str:
    position = str(config["position"]).strip().lower().replace("-", "_")
    mapping = {
        "random": "random",
        "bcos": "top1",
        "bcos_top1": "top1",
        "top1": "top1",
        "gradcam": "gradcam",
    }
    try:
        return mapping[position]
    except KeyError as exc:
        raise ValueError(f"Unsupported position: {config['position']}") from exc


def run_command(cmd: list[str], cwd: Path, log_path: Path, env: dict[str, str]) -> int:
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{now_iso()}] cwd={cwd}\n")
        log.write("[cmd] " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def write_manifest(path: Path, manifest: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def validate_output_contract(result_dir: Path) -> list[str]:
    missing: list[str] = []
    summary_path = result_dir / "summary.csv"
    success_events_path = result_dir / "success_events.csv"
    success_by_query_path = result_dir / "success_by_query.csv"
    for path in (summary_path, success_events_path, success_by_query_path):
        if not path.is_file():
            missing.append(str(path.name))
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            fieldnames = csv.DictReader(handle).fieldnames or []
        for name in (
            "first_success_query",
            "patch_position_y",
            "patch_position_x",
            "patch_position_h",
            "patch_position_w",
        ):
            if name not in fieldnames:
                missing.append(f"summary column {name}")
    return missing


def zip_result(zip_path: Path, run_dir: Path, result_dir: Path, extra_files: Iterable[Path]) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(result_dir.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, Path("outputs") / file_path.relative_to(result_dir))
        for file_path in extra_files:
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(run_dir))


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._") or "job"


def write_bundle_zip(zip_path: Path, run_dir: Path, extra_files: Iterable[Path]) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in extra_files:
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(run_dir))


def run_bundle(config: dict) -> int:
    bundle_id = str(config["job_id"])
    run_dir = Path("/kaggle/working") / bundle_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "bundle_manifest.json"
    zip_path = Path("/kaggle/working") / f"{bundle_id}_bundle_result.zip"
    repo = prepare_work_repo(config)
    runner = Path(__file__).resolve()
    started = time.time()
    results: list[dict[str, object]] = []

    manifest = {
        "job_id": bundle_id,
        "bundle": True,
        "bundle_size": len(config["bundle_jobs"]),
        "started_at": now_iso(),
        "repo": str(repo),
        "results": results,
    }
    write_manifest(manifest_path, manifest)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    for idx, sub_config in enumerate(config["bundle_jobs"], start=1):
        sub_config = dict(sub_config)
        sub_job_id = str(sub_config["job_id"])
        sub_config_path = run_dir / f"{idx:02d}_{safe_filename(sub_job_id)}_job_config.json"
        write_manifest(sub_config_path, sub_config)
        sub_started = time.time()
        sub_result = {
            "index": idx,
            "job_id": sub_job_id,
            "config": str(sub_config_path),
            "started_at": now_iso(),
            "result_zip": f"/kaggle/working/{sub_job_id}_result.zip",
        }
        results.append(sub_result)
        write_manifest(manifest_path, manifest)
        return_code = subprocess.call(
            [sys.executable, "-u", str(runner), "--config", str(sub_config_path)],
            cwd=str(repo),
            env=env,
        )
        sub_result["finished_at"] = now_iso()
        sub_result["elapsed_sec"] = round(time.time() - sub_started, 3)
        sub_result["return_code"] = return_code
        write_manifest(manifest_path, manifest)

    manifest["finished_at"] = now_iso()
    manifest["elapsed_sec"] = round(time.time() - started, 3)
    manifest["return_code"] = 0
    write_manifest(manifest_path, manifest)
    write_bundle_zip(zip_path, run_dir, [manifest_path, *sorted(run_dir.glob("*_job_config.json"))])
    print(f"Bundle result zip: {zip_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one CamoPatch Kaggle matrix job.")
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    args = parser.parse_args()

    config = load_config(args.config)
    if "bundle_jobs" in config:
        raise SystemExit(run_bundle(config))

    job_id = str(config["job_id"])
    run_dir = Path("/kaggle/working") / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir = DEFAULT_RESULT_ROOT / job_id
    result_dir.mkdir(parents=True, exist_ok=True)
    run_log = run_dir / "run.log"
    manifest_path = run_dir / "manifest.json"
    zip_path = Path("/kaggle/working") / f"{job_id}_result.zip"

    started = time.time()
    repo = prepare_work_repo(config)
    images_csv = prepare_images_csv(repo, config, run_dir)
    bcos_weights_dir = find_bcos_weights_dir()

    env = os.environ.copy()
    if bcos_weights_dir is not None:
        env["BCOS_WEIGHTS_DIR"] = str(bcos_weights_dir)
        env.setdefault("MODEL_WEIGHTS_DIR", str(bcos_weights_dir))
        env.setdefault("WEIGHTS_DIR", str(bcos_weights_dir))
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        "-u",
        str(repo / "CamoPatch" / "ConCamoPatchBatch.py"),
        "--images-csv",
        str(images_csv),
        "--save-root",
        str(result_dir),
        "--model",
        str(config["model"]),
        "--model_source",
        "bcos",
        "--device",
        str(config.get("device", "cuda")),
        "--image-batch-size",
        str(config.get("image_batch_size", 1000)),
        "--queries",
        str(config["queries"]),
        "--linf",
        str(config["linf"]),
        "--s",
        str(config["patch_size"]),
        "--position-rule",
        position_rule_for(config),
    ]
    if bool(config.get("fixed_position", True)):
        cmd.append("--fixed-position")
    if not bool(config.get("save_images", False)):
        cmd.append("--no_save_images")
    if "seed" in config:
        cmd.extend(["--seed", str(config["seed"])])
    if int(config.get("limit_images", 0) or 0) > 0:
        cmd.extend(["--limit-images", str(int(config["limit_images"]))])

    manifest = {
        "job_config": config,
        "job_id": job_id,
        "repo": str(repo),
        "images_csv": str(images_csv),
        "result_dir": str(result_dir),
        "result_zip": str(zip_path),
        "bcos_weights_dir": "" if bcos_weights_dir is None else str(bcos_weights_dir),
        "started_at": now_iso(),
    }
    write_manifest(manifest_path, manifest)

    return_code = run_command(cmd, cwd=repo, log_path=run_log, env=env)
    manifest["finished_at"] = now_iso()
    manifest["elapsed_sec"] = round(time.time() - started, 3)
    manifest["return_code"] = return_code
    missing_outputs = validate_output_contract(result_dir) if return_code == 0 else []
    if missing_outputs:
        manifest["failure_reason"] = "missing output contract: " + ", ".join(missing_outputs)
        manifest["return_code"] = 2
        return_code = 2
    write_manifest(manifest_path, manifest)

    zip_result(zip_path, run_dir, result_dir, [run_log, manifest_path])
    if return_code != 0:
        raise SystemExit(return_code)

    with zipfile.ZipFile(zip_path) as archive:
        bad_file = archive.testzip()
    if bad_file is not None:
        raise RuntimeError(f"Invalid result zip entry: {bad_file}")
    print(f"Result zip: {zip_path}")


if __name__ == "__main__":
    main()
