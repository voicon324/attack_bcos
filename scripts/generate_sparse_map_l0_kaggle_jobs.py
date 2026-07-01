#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET_SOURCES = [
    "hkhnhduy/attack-bcos-github",
    "hkhnhduy/weights-bcos",
    "sautkin/imagenet1kvalid",
]
COMPETITION_SOURCES = ["arc-prize-2026-arc-agi-3"]
MACHINE_SHAPE = "NvidiaRtxPro6000"


def build_job(args: argparse.Namespace) -> dict:
    temp_slug = "-".join(str(value).replace(".", "p") for value in args.temperatures)
    seed_slug = "-".join(str(value) for value in args.seeds)
    job_id = (
        f"sparse-map-l0-{args.model}-eps{args.eps_pixels}"
        f"-q{args.queries}-n{args.limit_images}-t{temp_slug}-seed{seed_slug}"
    )
    return {
        "job_id": job_id,
        "title": f"Sparse-RS L0 map ablation {args.model} eps {args.eps_pixels} q{args.queries} n{args.limit_images}",
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
        "timeout_minutes": args.timeout_minutes,
        "job_config": {
            "job_id": job_id,
            "attack": "sparse_rs_map_l0",
            "model_family": "bcos",
            "model": args.model,
            "patch_size": 0,
            "linf": f"L0/{args.eps_pixels}",
            "position": "map_ablation",
            "queries": args.queries,
            "eps_pixels": args.eps_pixels,
            "limit_images": args.limit_images,
            "images_csv": args.images_csv,
            "device": "cuda",
            "seeds": args.seeds,
            "temperatures": args.temperatures,
            "p_init": args.p_init,
            "rescale_schedule": True,
            "code_dataset_owner": "hkhnhduy",
            "code_dataset_slug": "attack-bcos-github",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one Sparse-RS L0 map-ablation Kaggle smoke job.")
    parser.add_argument("--output", type=Path, default=Path("kaggle/sparse_map_l0_smoke_jobs.json"))
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--eps-pixels", type=int, default=64)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--limit-images", type=int, default=100)
    parser.add_argument("--images-csv", default="data/used_images_1000.csv")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--temperatures", nargs="+", default=["4", "1", "0.25"])
    parser.add_argument("--p-init", type=float, default=0.8)
    parser.add_argument("--timeout-minutes", type=int, default=720)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = [build_job(args)]
    print(f"jobs={len(jobs)} output={args.output}")
    print(f"job_id={jobs[0]['job_id']}")
    if args.dry_run:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"jobs": jobs}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
