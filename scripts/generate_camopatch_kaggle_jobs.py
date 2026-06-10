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
PATCH_SIZES = [16, 8, 32]
LINFS = ["16/256", "32/256", "64/256"]
POSITIONS = ["random", "bcos_top1", "gradcam"]
VITC_MODELS = {"vitc_s", "vitc_b"}
DATASET_SOURCES = [
    "hkhnhduy/attack-bcos-github",
    "hkhnhduy/weights-bcos",
    "sautkin/imagenet1kvalid",
]
COMPETITION_SOURCES = ["nvidia-nemotron-model-reasoning-challenge"]
MACHINE_SHAPE = "NvidiaRtxPro6000"


def linf_slug(linf: str) -> str:
    return linf.replace("/", "_")


def iter_jobs() -> Iterable[dict]:
    for patch_size in PATCH_SIZES:
        for model in MODELS:
            for linf in LINFS:
                for position in POSITIONS:
                    if position == "gradcam" and model in VITC_MODELS:
                        continue
                    job_id = f"camopatch-bcos-{model}-s{patch_size}-linf{linf_slug(linf)}-{position}"
                    yield {
                        "job_id": job_id,
                        "title": f"CamoPatch B-cos {model} s{patch_size} {linf} {position}",
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
                            "attack": "camopatch",
                            "model_family": "bcos",
                            "model": model,
                            "patch_size": patch_size,
                            "linf": linf,
                            "position": position,
                            "queries": 10000,
                            "images_csv": "data/used_images_1000.csv",
                            "image_batch_size": 1000,
                            "device": "cuda",
                            "fixed_position": True,
                            "save_images": False,
                            "code_dataset_owner": "hkhnhduy",
                            "code_dataset_slug": "attack-bcos-github",
                        },
                    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CamoPatch B-cos Kaggle job config.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("kaggle/camopatch_jobs.json"),
        help="Where to write the generated job config.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing.")
    parser.add_argument("--smoke", action="store_true", help="Emit only the smoke-test job.")
    args = parser.parse_args()

    jobs = list(iter_jobs())
    if len(jobs) != 171:
        raise AssertionError(f"Expected 171 jobs, generated {len(jobs)}")
    if jobs[0]["job_config"]["patch_size"] != 16:
        raise AssertionError("Patch size 16 must be first in queue order.")
    if any(job["job_config"]["position"] == "gradcam" and job["job_config"]["model"] in VITC_MODELS for job in jobs):
        raise AssertionError("ViTC gradcam jobs must not be generated.")

    if args.smoke:
        jobs = [
            job for job in jobs
            if job["job_config"]["model"] == "resnet50"
            and job["job_config"]["patch_size"] == 16
            and job["job_config"]["linf"] == "16/256"
            and job["job_config"]["position"] == "bcos_top1"
        ]

    config = {"jobs": jobs}
    print(f"jobs={len(jobs)} output={args.output}")
    if args.dry_run:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
