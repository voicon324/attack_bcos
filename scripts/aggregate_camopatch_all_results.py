#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_CSV = ROOT / "data" / "used_images_1000.csv"
DEFAULT_IMAGE_ROOT = ROOT / "artifacts" / "data" / "imagenet"
DEFAULT_WEIGHTS_DIR = ROOT / "artifacts" / "model-weights" / "weights" / "bcos-imagenet"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "analysis" / "camopatch_all_results_latest"
DEFAULT_CLEAN_CSV = DEFAULT_OUTPUT_DIR / "clean_predictions_1000.csv"

DEFAULT_RUNS = [
    {
        "name": "fixed",
        "run_root": ROOT / "kaggle_runs_success_query_full",
        "jobs_config": ROOT / "kaggle" / "camopatch_jobs.json",
    },
    {
        "name": "movable",
        "run_root": ROOT / "kaggle_runs_movable_s16_linf64",
        "jobs_config": ROOT / "kaggle" / "camopatch_movable_s16_linf64_jobs.json",
    },
]

MODEL_ORDER = [
    "resnet18",
    "resnet50",
    "densenet121",
    "convnext_tiny",
    "convnext_base",
    "vitc_s",
    "vitc_b",
]

DETAIL_FIELDNAMES = [
    "attack",
    "run_name",
    "position_mode",
    "move_allowed",
    "job_id",
    "account",
    "url",
    "result_zip",
    "model",
    "model_source",
    "job_model",
    "patch_size",
    "linf",
    "eps_linf",
    "position",
    "position_rule",
    "queries",
    "image_index",
    "image_path",
    "local_image_path",
    "true_label",
    "clean_prediction",
    "clean_correct",
    "adversarial",
    "first_success_query",
    "final_prediction",
    "loc_y",
    "loc_x",
    "patch_position_y",
    "patch_position_x",
    "patch_position_h",
    "patch_position_w",
    "initial_loc_y",
    "initial_loc_x",
    "fixed_location",
    "location_source",
    "final_l2",
    "final_linf",
    "output_prefix",
    "elapsed_sec",
    "return_code",
    "done_at",
]

GROUP_KEYS = [
    "attack",
    "position_mode",
    "move_allowed",
    "model",
    "patch_size",
    "linf",
    "eps_linf",
    "position",
    "queries",
]

SUMMARY_FIELDNAMES = GROUP_KEYS + [
    "jobs",
    "images",
    "clean_correct_images",
    "successes",
    "success_rate",
    "median_first_success_query",
    "mean_first_success_query",
    "min_first_success_query",
    "max_first_success_query",
]

CURVE_FIELDNAMES = GROUP_KEYS + [
    "first_success_query",
    "new_successes",
    "cumulative_successes",
    "denominator_images",
    "success_rate",
    "image_indices",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate downloaded CamoPatch Kaggle result zips across fixed and movable runs, "
            "with all-image and clean-correct denominators."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-csv", type=Path, default=DEFAULT_IMAGE_CSV)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--clean-predictions-csv", type=Path, default=None)
    parser.add_argument("--compute-clean-predictions", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_ORDER,
        help="Models to include and/or compute clean predictions for. Defaults to the full B-cos model set.",
    )
    parser.add_argument(
        "--validate-zips",
        action="store_true",
        help="Run ZipFile.testzip() before reading each result. Slower, but checks every archive member.",
    )
    parser.add_argument(
        "--include-run",
        action="append",
        nargs=3,
        metavar=("NAME", "RUN_ROOT", "JOBS_CONFIG"),
        help="Override run list. May be repeated.",
    )
    return parser.parse_args()


def load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_zip_member(archive: zipfile.ZipFile, suffix: str) -> str | None:
    exact = suffix.lstrip("/")
    names = archive.namelist()
    if exact in names:
        return exact
    matches = sorted(name for name in names if name.endswith(suffix))
    return matches[0] if matches else None


def read_zip_csv(archive: zipfile.ZipFile, *suffixes: str) -> list[dict[str, str]]:
    member = None
    for suffix in suffixes:
        member = find_zip_member(archive, suffix)
        if member is not None:
            break
    if member is None:
        return []
    with archive.open(member) as handle:
        return list(csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8", newline="")))


def read_manifest(archive: zipfile.ZipFile) -> dict[str, Any]:
    member = find_zip_member(archive, "manifest.json")
    if member is None:
        return {}
    with archive.open(member) as handle:
        return json.loads(handle.read().decode("utf-8"))


def local_image_path(image_path: str, image_root: Path) -> Path:
    normalized = image_path.strip().replace("\\", "/")
    for marker in ("imagenet1kvalid/", "imagenet/"):
        if marker in normalized:
            return image_root / normalized.split(marker, 1)[1]
    path = Path(normalized)
    if path.is_absolute():
        return path
    return image_root / normalized


def infer_true_label(image_path: str) -> int:
    parent = Path(image_path.replace("\\", "/")).parent.name
    if not parent.isdigit():
        raise ValueError(f"Cannot infer true label from path: {image_path}")
    return int(parent)


def load_image_index(image_csv: Path, image_root: Path) -> list[dict[str, Any]]:
    rows = load_csv(image_csv)
    if not rows:
        raise ValueError(f"No rows in image CSV: {image_csv}")
    image_column = "image_path" if "image_path" in rows[0] else next(iter(rows[0]))
    indexed: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        image_path = row[image_column].strip()
        indexed.append(
            {
                "index": idx,
                "image_path": image_path,
                "local_image_path": str(local_image_path(image_path, image_root)),
                "true_label": infer_true_label(image_path),
            }
        )
    return indexed


def normalize_model_name(model: str) -> str:
    aliases = {
        "resnet_18": "resnet18",
        "resnet_50": "resnet50",
        "densenet_121": "densenet121",
        "vitc_s_patch1_14": "vitc_s",
        "vitc_b_patch1_14": "vitc_b",
    }
    return aliases.get(str(model).strip(), str(model).strip())


def as_int(value: Any, default: int | None = None) -> int | None:
    if value in ("", None):
        return default
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float | None = None) -> float | None:
    if value in ("", None):
        return default
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def bool_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if value in ("", None):
        return 0
    return 1 if str(value).strip().lower() in {"1", "true", "yes"} else 0


def compute_clean_predictions(
    image_csv: Path,
    image_root: Path,
    output_csv: Path,
    weights_dir: Path,
    device: str,
    batch_size: int,
    models: Iterable[str],
) -> Path:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    # Heavy ML imports stay inside this function so py_compile works in non-B-cos Python envs.
    import torch
    from PIL import Image
    from torchvision import transforms

    sys.path.insert(0, str(ROOT / "CamoPatch"))
    from ImageNetModels import ImageNetModel

    os.environ.setdefault("BCOS_WEIGHTS_DIR", str(weights_dir))
    os.environ.setdefault("MODEL_WEIGHTS_DIR", str(weights_dir.parent.parent))
    os.environ.setdefault("WEIGHTS_DIR", str(weights_dir.parent.parent))

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    image_rows = load_image_index(image_csv, image_root)
    missing = [row["local_image_path"] for row in image_rows if not Path(row["local_image_path"]).is_file()]
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} local image files. First missing:\n{preview}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "index", "image_path", "local_image_path", "true_label", "clean_prediction", "clean_correct"]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_name in models:
            print(f"clean_predictions: loading {model_name}", flush=True)
            model = ImageNetModel(model_name, device=device, model_source="bcos")
            for start in range(0, len(image_rows), batch_size):
                batch_rows = image_rows[start : start + batch_size]
                tensors = []
                for row in batch_rows:
                    with Image.open(row["local_image_path"]) as image:
                        tensors.append(transform(image.convert("RGB")))
                x = torch.stack(tensors, dim=0).to(device=model.device, dtype=torch.float32)
                logits = model.predict(x)
                preds = logits.argmax(dim=1).detach().cpu().tolist()
                out_rows = []
                for row, pred in zip(batch_rows, preds):
                    true_label = int(row["true_label"])
                    out_rows.append(
                        {
                            "model": model_name,
                            "index": int(row["index"]),
                            "image_path": row["image_path"],
                            "local_image_path": row["local_image_path"],
                            "true_label": true_label,
                            "clean_prediction": int(pred),
                            "clean_correct": int(int(pred) == true_label),
                        }
                    )
                writer.writerows(out_rows)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return output_csv


def load_clean_predictions(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    rows = load_csv(path)
    clean: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        model = normalize_model_name(row["model"])
        idx = as_int(row["index"])
        if idx is None:
            continue
        clean[(model, idx)] = row
    return clean


def configured_runs(args: argparse.Namespace) -> list[dict[str, Path | str]]:
    if args.include_run:
        return [
            {"name": name, "run_root": Path(run_root), "jobs_config": Path(jobs_config)}
            for name, run_root, jobs_config in args.include_run
        ]
    return DEFAULT_RUNS


def jobs_by_id(jobs_config: Path) -> dict[str, dict[str, Any]]:
    data = load_json(jobs_config, {"jobs": []})
    return {str(job.get("job_id")): job for job in data.get("jobs", [])}


def condition_from_row(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(key, "") for key in GROUP_KEYS)


def first_success_query(row: dict[str, Any]) -> int | None:
    query = as_int(row.get("first_success_query"))
    if query is None or query <= 0:
        return None
    return query


def aggregate_runs(
    runs: list[dict[str, Path | str]],
    clean_predictions: dict[tuple[str, int], dict[str, str]],
    image_csv: Path,
    image_root: Path,
    included_models: set[str] | None,
    validate_zips: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    image_rows = {row["index"]: row for row in load_image_index(image_csv, image_root)}
    detail_rows: list[dict[str, Any]] = []
    included_jobs: list[dict[str, Any]] = []
    skipped_jobs: list[dict[str, Any]] = []

    for run in runs:
        run_name = str(run["name"])
        run_root = Path(run["run_root"])
        jobs_config = Path(run["jobs_config"])
        state = load_json(run_root / "state.json", {"jobs": {}})
        configured = jobs_by_id(jobs_config)
        for job_id in sorted(state.get("jobs", {})):
            state_job = state["jobs"][job_id]
            status = str(state_job.get("status", ""))
            if status != "done":
                skipped_jobs.append({"run_name": run_name, "job_id": job_id, "status": status, "reason": "not done"})
                continue
            zip_value = str(state_job.get("result_zip", ""))
            zip_path = Path(zip_value) if zip_value else Path()
            if not zip_path.is_absolute():
                zip_path = ROOT / zip_path
            if not zip_path.is_file():
                skipped_jobs.append(
                    {"run_name": run_name, "job_id": job_id, "status": status, "reason": "result zip missing"}
                )
                continue
            try:
                with zipfile.ZipFile(zip_path) as archive:
                    if validate_zips:
                        bad_member = archive.testzip()
                        if bad_member:
                            skipped_jobs.append(
                                {
                                    "run_name": run_name,
                                    "job_id": job_id,
                                    "status": status,
                                    "reason": f"bad zip member: {bad_member}",
                                }
                            )
                            continue
                    manifest = read_manifest(archive)
                    rows = read_zip_csv(archive, "outputs/summary.csv", "summary.csv")
            except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
                skipped_jobs.append({"run_name": run_name, "job_id": job_id, "status": status, "reason": str(exc)})
                continue

            if not rows:
                skipped_jobs.append({"run_name": run_name, "job_id": job_id, "status": status, "reason": "no summary rows"})
                continue
            config = manifest.get("job_config") or configured.get(job_id, {}).get("job_config", {})
            fixed_position = bool(config.get("fixed_position", bool_int(rows[0].get("fixed_location", "1"))))
            move_allowed = int(not fixed_position)
            position_mode = "movable" if move_allowed else "fixed"
            job_model = normalize_model_name(str(config.get("model") or rows[0].get("model") or ""))
            if included_models is not None and job_model not in included_models:
                skipped_jobs.append(
                    {
                        "run_name": run_name,
                        "job_id": job_id,
                        "status": status,
                        "reason": f"model excluded: {job_model}",
                    }
                )
                continue
            job_position = str(config.get("position") or rows[0].get("position_rule") or "")
            attack = str(config.get("attack") or rows[0].get("attack") or "camopatch").strip() or "camopatch"
            job_linf = str(config.get("linf") or "")
            job_patch_size = as_int(config.get("patch_size") or rows[0].get("patch_size"), 0) or 0
            job_queries = as_int(config.get("queries") or rows[0].get("queries"), 0) or 0
            job_successes = 0

            for row in rows:
                image_index = as_int(row.get("index"))
                if image_index is None:
                    continue
                result_model = normalize_model_name(str(row.get("model") or job_model))
                clean = clean_predictions.get((job_model, image_index)) or clean_predictions.get((result_model, image_index))
                image_meta = image_rows.get(image_index, {})
                adversarial = bool_int(row.get("adversarial"))
                job_successes += adversarial
                detail_rows.append(
                    {
                        "attack": attack,
                        "run_name": run_name,
                        "position_mode": position_mode,
                        "move_allowed": move_allowed,
                        "job_id": job_id,
                        "account": state_job.get("account", ""),
                        "url": state_job.get("url", ""),
                        "result_zip": str(zip_path),
                        "model": job_model,
                        "model_source": row.get("model_source", ""),
                        "job_model": job_model,
                        "patch_size": job_patch_size,
                        "linf": job_linf,
                        "eps_linf": row.get("eps_linf", ""),
                        "position": job_position,
                        "position_rule": row.get("position_rule", ""),
                        "queries": job_queries,
                        "image_index": image_index,
                        "image_path": row.get("image_path", image_meta.get("image_path", "")),
                        "local_image_path": clean.get("local_image_path", image_meta.get("local_image_path", ""))
                        if clean
                        else image_meta.get("local_image_path", ""),
                        "true_label": as_int(row.get("true_label"), image_meta.get("true_label", "")),
                        "clean_prediction": clean.get("clean_prediction", "") if clean else "",
                        "clean_correct": bool_int(clean.get("clean_correct", "")) if clean else "",
                        "adversarial": adversarial,
                        "first_success_query": row.get("first_success_query", ""),
                        "final_prediction": row.get("final_prediction", ""),
                        "loc_y": row.get("loc_y", ""),
                        "loc_x": row.get("loc_x", ""),
                        "patch_position_y": row.get("patch_position_y", row.get("loc_y", "")),
                        "patch_position_x": row.get("patch_position_x", row.get("loc_x", "")),
                        "patch_position_h": row.get("patch_position_h", job_patch_size),
                        "patch_position_w": row.get("patch_position_w", job_patch_size),
                        "initial_loc_y": row.get("initial_loc_y", ""),
                        "initial_loc_x": row.get("initial_loc_x", ""),
                        "fixed_location": row.get("fixed_location", ""),
                        "location_source": row.get("location_source", ""),
                        "final_l2": row.get("final_l2", ""),
                        "final_linf": row.get("final_linf", ""),
                        "output_prefix": row.get("output_prefix", ""),
                        "elapsed_sec": manifest.get("elapsed_sec", ""),
                        "return_code": manifest.get("return_code", ""),
                        "done_at": state_job.get("done_at", ""),
                    }
                )
            included_jobs.append(
                {
                    "attack": attack,
                    "run_name": run_name,
                    "position_mode": position_mode,
                    "move_allowed": move_allowed,
                    "job_id": job_id,
                    "status": status,
                    "account": state_job.get("account", ""),
                    "url": state_job.get("url", ""),
                    "result_zip": str(zip_path),
                    "model": job_model,
                    "patch_size": job_patch_size,
                    "linf": job_linf,
                    "position": job_position,
                    "queries": job_queries,
                    "images": len(rows),
                    "successes": job_successes,
                    "success_rate": job_successes / len(rows) if rows else "",
                    "elapsed_sec": manifest.get("elapsed_sec", ""),
                    "return_code": manifest.get("return_code", ""),
                    "done_at": state_job.get("done_at", ""),
                }
            )

    return detail_rows, included_jobs, skipped_jobs


def summarize(rows: list[dict[str, Any]], *, clean_correct_only: bool) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if clean_correct_only and int(row.get("clean_correct") or 0) != 1:
            continue
        groups[condition_from_row(row)].append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        group_rows = groups[key]
        row = {field: value for field, value in zip(GROUP_KEYS, key)}
        success_queries = sorted(
            query for query in (first_success_query(item) for item in group_rows) if query is not None
        )
        successes = len(success_queries)
        images = len(group_rows)
        clean_correct_images = sum(int(item.get("clean_correct") or 0) for item in group_rows)
        row.update(
            {
                "jobs": len({item["job_id"] for item in group_rows}),
                "images": images,
                "clean_correct_images": clean_correct_images,
                "successes": successes,
                "success_rate": successes / images if images else "",
                "median_first_success_query": percentile(success_queries, 0.5),
                "mean_first_success_query": sum(success_queries) / len(success_queries) if success_queries else "",
                "min_first_success_query": min(success_queries) if success_queries else "",
                "max_first_success_query": max(success_queries) if success_queries else "",
            }
        )
        out.append(row)
    return out


def percentile(values: list[int], q: float) -> float | str:
    if not values:
        return ""
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * q
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    weight = pos - low
    return values[low] * (1.0 - weight) + values[high] * weight


def success_curve(rows: list[dict[str, Any]], *, clean_correct_only: bool) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if clean_correct_only and int(row.get("clean_correct") or 0) != 1:
            continue
        groups[condition_from_row(row)].append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        group_rows = groups[key]
        denominator = len(group_rows)
        event_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in group_rows:
            query = first_success_query(item)
            if query is not None:
                event_rows[query].append(item)
        cumulative = 0
        for query in sorted(event_rows):
            events = event_rows[query]
            cumulative += len(events)
            row = {field: value for field, value in zip(GROUP_KEYS, key)}
            row.update(
                {
                    "first_success_query": query,
                    "new_successes": len(events),
                    "cumulative_successes": cumulative,
                    "denominator_images": denominator,
                    "success_rate": cumulative / denominator if denominator else "",
                    "image_indices": ";".join(str(item["image_index"]) for item in events),
                }
            )
            out.append(row)
    return out


def zip_output_dir(output_dir: Path) -> Path:
    zip_path = output_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir.parent))
    return zip_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    clean_csv = (args.clean_predictions_csv or output_dir / "clean_predictions_1000.csv").resolve()
    runs = configured_runs(args)
    selected_models = [normalize_model_name(model) for model in args.models]
    included_models = set(selected_models)

    if args.compute_clean_predictions or not clean_csv.is_file():
        print(f"Computing clean predictions with conda/env Python: {sys.executable}", flush=True)
        compute_clean_predictions(
            image_csv=args.image_csv,
            image_root=args.image_root,
            output_csv=clean_csv,
            weights_dir=args.weights_dir,
            device=args.device,
            batch_size=args.batch_size,
            models=selected_models,
        )

    clean_predictions = load_clean_predictions(clean_csv)
    if not clean_predictions:
        raise ValueError(f"No clean predictions loaded from {clean_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_rows, included_jobs, skipped_jobs = aggregate_runs(
        runs=runs,
        clean_predictions=clean_predictions,
        image_csv=args.image_csv,
        image_root=args.image_root,
        included_models=included_models,
        validate_zips=args.validate_zips,
    )
    if not detail_rows:
        raise ValueError("No detail rows were aggregated.")

    all_detail = output_dir / "combined_image_results_all.csv"
    clean_detail = output_dir / "combined_image_results_clean_correct.csv"
    write_csv(all_detail, detail_rows, DETAIL_FIELDNAMES)
    write_csv(
        clean_detail,
        [row for row in detail_rows if int(row.get("clean_correct") or 0) == 1],
        DETAIL_FIELDNAMES,
    )
    write_csv(output_dir / "included_jobs.csv", included_jobs)
    write_csv(output_dir / "skipped_jobs.csv", skipped_jobs)
    write_csv(output_dir / "summary_all_images.csv", summarize(detail_rows, clean_correct_only=False), SUMMARY_FIELDNAMES)
    write_csv(
        output_dir / "summary_clean_correct.csv",
        summarize(detail_rows, clean_correct_only=True),
        SUMMARY_FIELDNAMES,
    )
    write_csv(
        output_dir / "success_by_query_all_images.csv",
        success_curve(detail_rows, clean_correct_only=False),
        CURVE_FIELDNAMES,
    )
    write_csv(
        output_dir / "success_by_query_clean_correct.csv",
        success_curve(detail_rows, clean_correct_only=True),
        CURVE_FIELDNAMES,
    )

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = {
        "generated_at": timestamp,
        "script": str(Path(__file__).relative_to(ROOT)),
        "python": sys.executable,
        "image_csv": str(args.image_csv),
        "image_root": str(args.image_root),
        "clean_predictions_csv": str(clean_csv),
        "included_models": selected_models,
        "runs": [
            {
                "name": str(run["name"]),
                "run_root": str(run["run_root"]),
                "jobs_config": str(run["jobs_config"]),
            }
            for run in runs
        ],
        "included_jobs": len(included_jobs),
        "skipped_jobs": len(skipped_jobs),
        "image_result_rows_all": len(detail_rows),
        "image_result_rows_clean_correct": sum(int(row.get("clean_correct") or 0) == 1 for row in detail_rows),
        "notes": [
            "Clean predictions are recomputed from the original clean images, then joined by model and one-based CSV index.",
            "Only the fixed full run and movable s16 linf64 run are included by default; smoke, fake, failed, and old duplicate runs are excluded.",
            "success_by_query files contain cumulative success rates at first-success query values.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    clean_copy = output_dir / "clean_predictions_1000.csv"
    if clean_csv.resolve() != clean_copy.resolve():
        shutil.copy2(clean_csv, clean_copy)
    zip_path = zip_output_dir(output_dir)

    print(f"output_dir={output_dir}")
    print(f"zip={zip_path}")
    print(f"included_jobs={len(included_jobs)}")
    print(f"image_result_rows_all={len(detail_rows)}")
    print(f"image_result_rows_clean_correct={manifest['image_result_rows_clean_correct']}")


if __name__ == "__main__":
    main()
