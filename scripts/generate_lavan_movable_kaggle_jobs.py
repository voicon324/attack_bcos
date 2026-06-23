#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


MODELS = [
    "resnet18",
    "resnet50",
    "densenet121",
    "convnext_tiny",
    "convnext_base",
    "vitc_s",
    "vitc_b",
]
PATCH_SIZE = 16
LINF = "64/256"
POSITIONS = ["random", "bcos_top1"]
DATASET_SOURCES = [
    "hkhnhduy/attack-bcos-github",
    "hkhnhduy/weights-bcos",
    "sautkin/imagenet1kvalid",
]
COMPETITION_SOURCES = ["arc-prize-2026-arc-agi-3"]
MACHINE_SHAPE = "NvidiaRtxPro6000"


def linf_slug(linf: str) -> str:
    return linf.replace("/", "_")


def iter_jobs(queries: int) -> Iterable[dict]:
    for model in MODELS:
        for position in POSITIONS:
            job_id = (
                f"lavan-bcos-movable-{model}-s{PATCH_SIZE}-"
                f"linf{linf_slug(LINF)}-init-{position}"
            )
            yield {
                "job_id": job_id,
                "title": f"LaVAN B-cos movable {model} s{PATCH_SIZE} {LINF} init {position}",
                "template_dir": "kaggle/camopatch_job_template",
                "code_file": "run_kernel.py",
                "kernel_type": "script",
                "language": "python",
                "enable_gpu": True,
                "enable_internet": False,
                "machine_shape": MACHINE_SHAPE,
                "is_private": True,
                "dataset_sources": DATASET_SOURCES,
                "competition_sources": COMPETITION_SOURCES,
                "output_pattern": ".*(zip|log|json|csv)$",
                "expected_zip": f"{job_id}_result.zip",
                "timeout_minutes": 720,
                "job_config": {
                    "job_id": job_id,
                    "attack": "lavan",
                    "model_family": "bcos",
                    "model": model,
                    "patch_size": PATCH_SIZE,
                    "linf": LINF,
                    "position": position,
                    "queries": queries,
                    "images_csv": "data/used_images_1000.csv",
                    "image_batch_size": 16,
                    "device": "cuda",
                    "fixed_position": False,
                    "save_images": False,
                    "step_size": "1/256",
                    "gradient_mode": "sign",
                    "patch_init": "random_linf",
                    "code_dataset_owner": "hkhnhduy",
                    "code_dataset_slug": "attack-bcos-github",
                    "experiment": "lavan_movable_s16_linf64",
                },
            }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate movable-position LaVAN B-cos Kaggle jobs."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("kaggle/lavan_movable_s16_linf64_jobs.json"),
        help="Where to write the generated job config.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=500,
        help="LaVAN optimization iterations/evaluations per image.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing.")
    args = parser.parse_args()

    if args.queries <= 0:
        parser.error("--queries must be > 0")

    jobs = list(iter_jobs(args.queries))
    if len(jobs) != len(MODELS) * len(POSITIONS):
        raise AssertionError(f"Expected 14 jobs, generated {len(jobs)}")
    if any(job["job_config"]["patch_size"] != 16 for job in jobs):
        raise AssertionError("All movable jobs must use patch size 16.")
    if any(job["job_config"]["linf"] != "64/256" for job in jobs):
        raise AssertionError("All movable jobs must use L_inf 64/256.")
    if any(job["job_config"]["fixed_position"] for job in jobs):
        raise AssertionError("Movable jobs must set fixed_position=false.")

    config = {"jobs": jobs}
    print(f"jobs={len(jobs)} queries={args.queries} output={args.output}")
    if args.dry_run:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
