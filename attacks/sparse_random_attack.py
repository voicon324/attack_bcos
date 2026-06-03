"""
Random sparse baseline for B-cos models.

Each attempt samples K unchanged pixels uniformly at random, builds a small set
of value candidates for each sampled pixel, evaluates every candidate with a
real model forward pass, and applies the candidate that maximizes the fixed
runner-up-minus-top-1 margin.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torchvision import models as tv_models

import sparse_attack as explain_sparse
from bcos_es_patch import load_rgb_image


RESULT_DIR = explain_sparse.RESULT_DIR
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ForwardFn = Callable[[Tensor], Tensor]


def normalize_imagenet(x_rgb: Tensor, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> Tensor:
    mean_tensor = x_rgb.new_tensor(mean).view(1, 3, 1, 1)
    std_tensor = x_rgb.new_tensor(std).view(1, 3, 1, 1)
    return (x_rgb - mean_tensor) / std_tensor


def resolve_torchvision_weights(model_name: str) -> Any:
    weights_enum = tv_models.get_model_weights(model_name)
    weight_v1 = getattr(weights_enum, "IMAGENET1K_V1", None)
    if weight_v1 is not None:
        return weight_v1
    return weights_enum.DEFAULT


def load_torchvision_model(
    model_name: str,
    device: torch.device,
    channels_last: bool,
) -> Tuple[torch.nn.Module, Tuple[float, float, float], Tuple[float, float, float]]:
    weights = resolve_torchvision_weights(model_name)
    weight_transforms = weights.transforms()
    mean = tuple(float(v) for v in getattr(weight_transforms, "mean", IMAGENET_MEAN))
    std = tuple(float(v) for v in getattr(weight_transforms, "std", IMAGENET_STD))
    model = tv_models.get_model(model_name, weights=weights).to(device).eval()
    if channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model, mean, std


def model_forward_torchvision(
    model: torch.nn.Module,
    x_rgb: Tensor,
    channels_last: bool,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> Tensor:
    x_norm = normalize_imagenet(x_rgb, mean=mean, std=std)
    x_norm = explain_sparse.maybe_channels_last(x_norm, channels_last)
    return model(x_norm)


def build_random_output_dir(output_root: Path, image_path: str, image_index: Optional[int]) -> Path:
    image_file = Path(image_path)
    parts: List[str] = []
    if image_index is not None:
        parts.append(f"{image_index:03d}")
    if image_file.parent.name:
        parts.append(image_file.parent.name)
    parts.append(image_file.stem)
    safe_name = "_".join(part for part in parts if part)
    return output_root / f"{safe_name}_random_sparse"


def pixel_value_candidates(
    old_values: Tuple[float, float, float],
    value_set: str,
    rng: random.Random,
) -> List[Tuple[float, float, float]]:
    if value_set == "black-white":
        raw_values = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
    elif value_set == "rgb-corners":
        raw_values = [tuple(float(v) for v in values) for values in itertools.product((0, 1), repeat=3)]
    elif value_set == "invert":
        raw_values = [tuple(1.0 - value for value in old_values)]
    elif value_set == "random-corner":
        raw_values = [tuple(float(rng.randrange(2)) for _ in range(3))]
    else:
        raise ValueError(f"Unsupported value set '{value_set}'.")

    candidates: List[Tuple[float, float, float]] = []
    for values in raw_values:
        if all(abs(values[idx] - old_values[idx]) < 1e-12 for idx in range(3)):
            continue
        candidates.append(values)
    return candidates


def build_random_candidates(
    x_rgb: Tensor,
    changed_mask: Tensor,
    random_pixels: int,
    value_set: str,
    rng: random.Random,
) -> List[explain_sparse.SparseCandidate]:
    available = torch.nonzero(~changed_mask, as_tuple=False).detach().cpu().tolist()
    if not available:
        return []

    sample_count = min(random_pixels, len(available))
    sampled_positions = rng.sample(available, sample_count)
    candidates: List[explain_sparse.SparseCandidate] = []
    for y_raw, x_raw in sampled_positions:
        y = int(y_raw)
        x = int(x_raw)
        old_values = tuple(float(v) for v in x_rgb[0, :, y, x].detach().cpu().tolist())
        for new_values in pixel_value_candidates(old_values, value_set=value_set, rng=rng):
            candidates.append(
                explain_sparse.SparseCandidate(
                    estimated_gain=0.0,
                    y=y,
                    x=x,
                    channel=-1,
                    new_values=new_values,
                )
            )
    return candidates


def score_candidates_with_forward(
    forward_fn: ForwardFn,
    x_rgb: Tensor,
    candidates: Sequence[explain_sparse.SparseCandidate],
    top1_class: int,
    top2_class: int,
    candidate_batch_size: int,
) -> Tuple[explain_sparse.SparseCandidate, Tensor]:
    if not candidates:
        raise ValueError("Cannot score an empty candidate list.")

    margins: List[Tensor] = []
    batch_size = max(1, candidate_batch_size)
    with torch.inference_mode():
        for start in range(0, len(candidates), batch_size):
            chunk = candidates[start:start + batch_size]
            x_batch = explain_sparse.apply_sparse_candidates_batch(x_rgb, chunk)
            outputs = forward_fn(x_batch)
            margins.append(explain_sparse.margin_for_classes(outputs, top1_class, top2_class).detach())
    all_margins = torch.cat(margins, dim=0)
    best_idx = int(all_margins.argmax().item())
    return candidates[best_idx], all_margins


def save_random_steps_csv(steps: Sequence[explain_sparse.SparseStep], path: Path) -> None:
    fieldnames = [
        "step",
        "y",
        "x",
        "old_r",
        "old_g",
        "old_b",
        "new_r",
        "new_g",
        "new_b",
        "verified_margin_gain",
        "margin_before",
        "margin_after",
        "pred_after",
        "top1_logit_after",
        "top2_logit_after",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step in steps:
            writer.writerow(
                {
                    "step": step.step,
                    "y": step.y,
                    "x": step.x,
                    "old_r": step.old_values[0],
                    "old_g": step.old_values[1],
                    "old_b": step.old_values[2],
                    "new_r": step.new_values[0],
                    "new_g": step.new_values[1],
                    "new_b": step.new_values[2],
                    "verified_margin_gain": step.estimated_gain,
                    "margin_before": step.margin_before,
                    "margin_after": step.margin_after,
                    "pred_after": step.pred_after,
                    "top1_logit_after": step.top1_logit_after,
                    "top2_logit_after": step.top2_logit_after,
                }
            )


def run_random_sparse_attack(
    model: torch.nn.Module,
    forward_fn: ForwardFn,
    x_rgb: Tensor,
    max_pixels: int,
    random_pixels: int,
    value_set: str,
    candidate_batch_size: int,
    max_attempts: int,
    allow_non_improving: bool,
    channels_last: bool,
    rng: random.Random,
    verbose: bool,
) -> explain_sparse.SparseAttackResult:
    if x_rgb.shape[0] != 1:
        raise ValueError("run_random_sparse_attack expects a single image batch.")
    if max_pixels <= 0:
        raise ValueError("--max-pixels must be > 0.")

    x_adv = x_rgb.clone()
    _, _, height, width = x_adv.shape
    changed_mask = torch.zeros((height, width), dtype=torch.bool, device=x_adv.device)

    with torch.inference_mode():
        orig_outputs = forward_fn(x_adv)
        top1_class = int(orig_outputs.argmax(dim=1).item())
        top2_class = explain_sparse.resolve_runner_up_class(orig_outputs, top1_class)
        original_top1_logit = float(orig_outputs[0, top1_class].item())
        original_top2_logit = float(orig_outputs[0, top2_class].item())
        original_margin = float((orig_outputs[0, top2_class] - orig_outputs[0, top1_class]).item())
        original_top1_prob = explain_sparse.probability_value(model, orig_outputs, top1_class)
        original_top2_prob = explain_sparse.probability_value(model, orig_outputs, top2_class)

    current_margin = original_margin
    steps: List[explain_sparse.SparseStep] = []
    attempts = 0

    while len(steps) < max_pixels and attempts < max_attempts:
        attempts += 1
        candidates = build_random_candidates(
            x_rgb=x_adv,
            changed_mask=changed_mask,
            random_pixels=random_pixels,
            value_set=value_set,
            rng=rng,
        )
        if not candidates:
            if verbose:
                print("  No random candidates left.")
            break

        margin_before = current_margin
        best_candidate, candidate_margins = score_candidates_with_forward(
            forward_fn=forward_fn,
            x_rgb=x_adv,
            candidates=candidates,
            top1_class=top1_class,
            top2_class=top2_class,
            candidate_batch_size=candidate_batch_size,
        )
        best_margin = float(candidate_margins.max().item())
        x_candidate = explain_sparse.apply_sparse_candidate(x_adv, best_candidate)
        with torch.inference_mode():
            candidate_outputs = forward_fn(x_candidate)
            pred_after = int(candidate_outputs.argmax(dim=1).item())
            top1_logit_after = float(candidate_outputs[0, top1_class].item())
            top2_logit_after = float(candidate_outputs[0, top2_class].item())
            actual_margin = float((candidate_outputs[0, top2_class] - candidate_outputs[0, top1_class]).item())
        verified_gain = actual_margin - current_margin

        if verified_gain <= 0.0 and not allow_non_improving:
            if verbose:
                print(
                    f"  Attempt {attempts:03d}: no improving random candidate "
                    f"({current_margin:.6f} -> {actual_margin:.6f}; "
                    f"batch best {best_margin:.6f})."
                )
            continue

        old_values = tuple(float(v) for v in x_adv[0, :, best_candidate.y, best_candidate.x].detach().cpu().tolist())
        x_adv = x_candidate
        changed_mask[best_candidate.y, best_candidate.x] = True
        current_margin = actual_margin

        step_idx = len(steps) + 1
        steps.append(
            explain_sparse.SparseStep(
                step=step_idx,
                y=best_candidate.y,
                x=best_candidate.x,
                channel=-1,
                old_values=old_values,
                new_values=best_candidate.new_values,
                estimated_gain=verified_gain,
                margin_before=margin_before,
                margin_after=current_margin,
                pred_after=pred_after,
                top1_logit_after=top1_logit_after,
                top2_logit_after=top2_logit_after,
            )
        )

        if verbose:
            print(
                f"  Step {step_idx:03d}/{max_pixels:03d} | attempt={attempts:03d} | "
                f"pixel=({best_candidate.y},{best_candidate.x}) | "
                f"gain={verified_gain:.4f} | margin={current_margin:.4f} | pred={pred_after}"
            )

        if pred_after != top1_class:
            break

    with torch.inference_mode():
        final_outputs = forward_fn(x_adv)
        final_class = int(final_outputs.argmax(dim=1).item())
        final_top1_logit = float(final_outputs[0, top1_class].item())
        final_top2_logit = float(final_outputs[0, top2_class].item())
        final_margin = float((final_outputs[0, top2_class] - final_outputs[0, top1_class]).item())
        final_top1_prob = explain_sparse.probability_value(model, final_outputs, top1_class)
        final_top2_prob = explain_sparse.probability_value(model, final_outputs, top2_class)

    return explain_sparse.SparseAttackResult(
        x_adv=x_adv,
        original_class=top1_class,
        runner_up_class=top2_class,
        final_class=final_class,
        original_top1_logit=original_top1_logit,
        original_top2_logit=original_top2_logit,
        final_top1_logit=final_top1_logit,
        final_top2_logit=final_top2_logit,
        original_top1_prob=original_top1_prob,
        original_top2_prob=original_top2_prob,
        final_top1_prob=final_top1_prob,
        final_top2_prob=final_top2_prob,
        original_margin=original_margin,
        final_margin=final_margin,
        success=final_class != top1_class,
        steps=steps,
    )


def process_single_image(
    model: torch.nn.Module,
    image_path: str,
    image_index: Optional[int],
    output_root: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    x_rgb, resolved_image_path = load_rgb_image(image_path, device=args.device_obj)
    x_rgb = explain_sparse.maybe_channels_last(x_rgb, args.channels_last)
    out_dir = build_random_output_dir(output_root, resolved_image_path, image_index)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nImage: {resolved_image_path}")
    result = run_random_sparse_attack(
        model=model,
        forward_fn=args.forward_fn,
        x_rgb=x_rgb,
        max_pixels=args.max_pixels,
        random_pixels=args.random_pixels,
        value_set=args.value_set,
        candidate_batch_size=args.candidate_batch_size,
        max_attempts=args.max_attempts,
        allow_non_improving=args.allow_non_improving,
        channels_last=args.channels_last,
        rng=args.rng,
        verbose=args.verbose,
    )

    if args.save_images:
        import torchvision.utils as vutils

        vutils.save_image(x_rgb, str(out_dir / "before.png"))
        vutils.save_image(result.x_adv, str(out_dir / "after.png"))
        explain_sparse.save_changed_overlay(x_rgb, result, out_dir / "changed_pixels.png")
    save_random_steps_csv(result.steps, out_dir / "steps.csv")

    print(
        "  "
        f"top1={result.original_class} runner_up={result.runner_up_class} "
        f"final={result.final_class} success={int(result.success)}"
    )
    print(
        "  "
        f"margin {result.original_margin:.4f} -> {result.final_margin:.4f} | "
        f"changed_pixels={explain_sparse.count_changed_pixels(result.steps)} steps={len(result.steps)}"
    )

    return {
        "index": image_index,
        "image_path": resolved_image_path,
        "output_dir": str(out_dir),
        "original_class": result.original_class,
        "runner_up_class": result.runner_up_class,
        "final_class": result.final_class,
        "success": int(result.success),
        "steps": len(result.steps),
        "changed_pixels": explain_sparse.count_changed_pixels(result.steps),
        "original_top1_logit": result.original_top1_logit,
        "original_top2_logit": result.original_top2_logit,
        "final_top1_logit": result.final_top1_logit,
        "final_top2_logit": result.final_top2_logit,
        "original_top1_prob": result.original_top1_prob,
        "original_top2_prob": result.original_top2_prob,
        "final_top1_prob": result.final_top1_prob,
        "final_top2_prob": result.final_top2_prob,
        "original_margin": result.original_margin,
        "final_margin": result.final_margin,
        "random_pixels": args.random_pixels,
        "value_set": args.value_set,
        "model_family": args.model_family,
        "model": args.model,
    }


def write_summary(results: Sequence[Dict[str, Any]], summary_path: Path) -> None:
    if not results:
        return
    fieldnames = [
        "index",
        "image_path",
        "output_dir",
        "original_class",
        "runner_up_class",
        "final_class",
        "success",
        "steps",
        "changed_pixels",
        "original_top1_logit",
        "original_top2_logit",
        "final_top1_logit",
        "final_top2_logit",
        "original_top1_prob",
        "original_top2_prob",
        "final_top1_prob",
        "final_top2_prob",
        "original_margin",
        "final_margin",
        "random_pixels",
        "value_set",
        "model_family",
        "model",
    ]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random sparse pixel baseline for B-cos or torchvision models.",
    )
    parser.add_argument("--image", type=str, default=None, help="Path to one input image")
    parser.add_argument("--images-csv", type=str, default=None, help="CSV containing an image_path column")
    parser.add_argument(
        "--model-family",
        choices=("bcos", "torchvision"),
        default="bcos",
        help="Attack a B-cos model or a normal torchvision ImageNet model",
    )
    parser.add_argument("--model", type=str, default="resnet50", help="Model name")
    parser.add_argument("--max-pixels", type=int, default=20, help="Maximum number of successful sparse edits")
    parser.add_argument("--random-pixels", type=int, default=32, help="Random unchanged pixels sampled per attempt")
    parser.add_argument(
        "--value-set",
        choices=("rgb-corners", "black-white", "invert", "random-corner"),
        default="rgb-corners",
        help="Pixel values tested for each sampled random pixel",
    )
    parser.add_argument("--candidate-batch-size", type=int, default=128, help="Candidate images per forward pass")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Maximum random sampling attempts. 0 = max_pixels * 20.",
    )
    parser.add_argument(
        "--allow-non-improving",
        action="store_true",
        help="Apply the best random candidate even when it does not improve the fixed top2-top1 margin",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-images", action="store_true", help="Save before/after PNGs and changed-pixel overlay")
    parser.add_argument("--verbose", action="store_true", help="Print per-attempt/per-step progress")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULT_DIR / "sparse_random_attack_outputs"),
        help="Root folder for generated outputs",
    )
    args = parser.parse_args()

    if args.image and args.images_csv:
        parser.error("Use either --image or --images-csv, not both.")
    if not args.image and not args.images_csv:
        parser.error("Provide --image or --images-csv.")
    if args.max_pixels <= 0:
        parser.error("--max-pixels must be > 0.")
    if args.random_pixels <= 0:
        parser.error("--random-pixels must be > 0.")
    if args.candidate_batch_size <= 0:
        parser.error("--candidate-batch-size must be > 0.")
    if args.max_attempts < 0:
        parser.error("--max-attempts must be >= 0.")
    if args.max_attempts == 0:
        args.max_attempts = args.max_pixels * 20
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.rng = random.Random(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    explain_sparse.configure_fast_runtime(device)
    args.device_obj = device
    args.channels_last = device.type == "cuda"

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Random Sparse Attack")
    print("=" * 72)
    print(f"  Device       : {device}")
    print(f"  Model family : {args.model_family}")
    print(f"  Model        : {args.model}")
    print(f"  Max pixels   : {args.max_pixels}")
    print(f"  Random pixels: {args.random_pixels}")
    print(f"  Value set    : {args.value_set}")
    print(f"  Max attempts : {args.max_attempts}")
    print(f"  Output root  : {output_root}")

    if args.model_family == "bcos":
        print("\nLoading B-cos model ...")
        model_fn = getattr(explain_sparse.bcos.pretrained, args.model)
        model = model_fn(pretrained=True).to(device).eval()
        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)
        args.forward_fn = lambda x_batch: explain_sparse.model_forward_bcos(
            model,
            x_batch,
            channels_last=args.channels_last,
        )
    else:
        print("\nLoading torchvision model ...")
        model, mean, std = load_torchvision_model(
            args.model,
            device=device,
            channels_last=args.channels_last,
        )
        args.forward_fn = lambda x_batch: model_forward_torchvision(
            model,
            x_batch,
            channels_last=args.channels_last,
            mean=mean,
            std=std,
        )
        print(f"  Normalize    : mean={mean}, std={std}")

    if args.images_csv:
        image_items = explain_sparse.load_image_paths_from_csv(Path(args.images_csv))
        print(f"\nLoaded {len(image_items)} image paths from {args.images_csv}")
    else:
        image_items = [(None, args.image)]

    results: List[Dict[str, Any]] = []
    for item_num, (image_index, image_path) in enumerate(image_items, start=1):
        print("\n" + "-" * 72)
        print(f"Image {item_num}/{len(image_items)}")
        print("-" * 72)
        results.append(
            process_single_image(
                model=model,
                image_path=image_path,
                image_index=image_index,
                output_root=output_root,
                args=args,
            )
        )

    summary_path = output_root / "summary.csv"
    write_summary(results, summary_path)
    successes = sum(int(row["success"]) for row in results)
    print(f"\nSummary saved to {summary_path}")
    print(f"Successful class changes: {successes}/{len(results)}")
    print("\nDone")


if __name__ == "__main__":
    main()
