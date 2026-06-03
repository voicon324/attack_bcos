"""
Greedy sparse attack for B-cos models.

The attack uses B-cos explanations as a local linear model. For a B-cos input
z = [x, 1 - x], changing an RGB value x by dx changes a class logit by

    delta_y ~= (W_rgb - W_inv) * dx

where W is `dynamic_linear_weights` from `model.explain()`. At each step this
script chooses sparse pixel edits that maximize the estimated increase of the
runner-up class logit minus the decrease of the original top-1 class logit, then
optionally verifies the top candidates with a real model forward pass.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT
RESULT_DIR = SCRIPT_DIR / "artifacts" / "outputs" / "result"
sys.path.insert(0, str(SCRIPT_DIR))
LOCAL_BCOS_DIR = SCRIPT_DIR / "B-cos-v2"
if LOCAL_BCOS_DIR.is_dir():
    sys.path.insert(0, str(LOCAL_BCOS_DIR))
elif Path("/kaggle/working/B-cos-v2").is_dir():
    sys.path.insert(0, "/kaggle/working/B-cos-v2")

import bcos
from bcos_es_patch import extract_attribution, load_rgb_image, to_bcos_input


@dataclass(frozen=True)
class SparseCandidate:
    estimated_gain: float
    y: int
    x: int
    channel: int
    new_values: Tuple[float, float, float]


@dataclass(frozen=True)
class SparseStep:
    step: int
    y: int
    x: int
    channel: int
    old_values: Tuple[float, float, float]
    new_values: Tuple[float, float, float]
    estimated_gain: float
    margin_before: float
    margin_after: float
    pred_after: int
    top1_logit_after: float
    top2_logit_after: float


@dataclass(frozen=True)
class SparseAttackResult:
    x_adv: Tensor
    original_class: int
    runner_up_class: int
    final_class: int
    original_top1_logit: float
    original_top2_logit: float
    final_top1_logit: float
    final_top2_logit: float
    original_top1_prob: float
    original_top2_prob: float
    final_top1_prob: float
    final_top2_prob: float
    original_margin: float
    final_margin: float
    success: bool
    steps: List[SparseStep]


def configure_fast_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def maybe_channels_last(x: Tensor, enabled: bool) -> Tensor:
    if enabled and x.dim() == 4 and x.device.type == "cuda":
        return x.contiguous(memory_format=torch.channels_last)
    return x


def model_forward_bcos(
    model: torch.nn.Module,
    x_rgb: Tensor,
    channels_last: bool,
) -> Tensor:
    x_bcos = to_bcos_input(x_rgb)
    x_bcos = maybe_channels_last(x_bcos, channels_last)
    return model(x_bcos)


def probability_value(model: torch.nn.Module, outputs: Tensor, class_idx: int) -> float:
    to_probabilities = getattr(model, "to_probabilities", None)
    if to_probabilities is None:
        probs = torch.softmax(outputs, dim=1)
    else:
        probs = to_probabilities(outputs)
    return float(probs[0, class_idx].item())


def resolve_runner_up_class(outputs: Tensor, top1_class: int) -> int:
    masked = outputs.clone()
    masked[0, top1_class] = -float("inf")
    return int(masked.argmax(dim=1).item())


def margin_for_classes(outputs: Tensor, top1_class: int, top2_class: int) -> Tensor:
    return outputs[:, top2_class] - outputs[:, top1_class]


def effective_rgb_weights(dynamic_linear_weights: Tensor) -> Tensor:
    weights = dynamic_linear_weights
    if weights.dim() != 4 or weights.shape[0] != 1 or weights.shape[1] != 6:
        raise ValueError(
            "Expected dynamic_linear_weights with shape [1, 6, H, W], "
            f"got {tuple(weights.shape)}."
        )
    return weights[0, :3] - weights[0, 3:6]


def desired_values_for_direction(
    x_rgb: Tensor,
    direction: Tensor,
    value_mode: str,
    epsilon: float,
) -> Tensor:
    current = x_rgb[0]
    if value_mode == "corners":
        return torch.where(direction >= 0, torch.ones_like(current), torch.zeros_like(current))
    if value_mode == "linf":
        return (current + epsilon * direction.sign()).clamp(0.0, 1.0)
    raise ValueError(f"Unsupported value mode '{value_mode}'.")


def build_sparse_candidates(
    x_rgb: Tensor,
    attr_top1: Dict[str, Any],
    attr_top2: Dict[str, Any],
    changed_mask: Tensor,
    candidate_topk: int,
    granularity: str,
    value_mode: str,
    epsilon: float,
) -> List[SparseCandidate]:
    top1_eff = effective_rgb_weights(attr_top1["dynamic_linear_weights"])
    top2_eff = effective_rgb_weights(attr_top2["dynamic_linear_weights"])
    direction = top2_eff - top1_eff
    desired_values = desired_values_for_direction(
        x_rgb=x_rgb,
        direction=direction,
        value_mode=value_mode,
        epsilon=epsilon,
    )
    deltas = desired_values - x_rgb[0]
    channel_gains = direction * deltas

    if granularity == "pixel":
        scores = channel_gains.sum(dim=0)
        scores = scores.masked_fill(changed_mask, -float("inf"))
        flat_scores = scores.flatten()
        available = int((~changed_mask).sum().item())
        if available <= 0:
            return []
        topk = min(candidate_topk, available)
        values, flat_indices = torch.topk(flat_scores, k=topk, largest=True, sorted=True)
        candidates: List[SparseCandidate] = []
        _, height, width = x_rgb.shape[1:]
        for score, flat_idx in zip(values.tolist(), flat_indices.tolist()):
            if not np.isfinite(score):
                continue
            y, x = divmod(int(flat_idx), width)
            new_values = tuple(float(v) for v in desired_values[:, y, x].detach().cpu().tolist())
            candidates.append(
                SparseCandidate(
                    estimated_gain=float(score),
                    y=y,
                    x=x,
                    channel=-1,
                    new_values=new_values,
                )
            )
        return candidates

    if granularity == "channel":
        scores = channel_gains.masked_fill(changed_mask, -float("inf"))
        flat_scores = scores.flatten()
        available = int((~changed_mask).sum().item())
        if available <= 0:
            return []
        topk = min(candidate_topk, available)
        values, flat_indices = torch.topk(flat_scores, k=topk, largest=True, sorted=True)
        candidates = []
        _, height, width = x_rgb.shape[1:]
        for score, flat_idx in zip(values.tolist(), flat_indices.tolist()):
            if not np.isfinite(score):
                continue
            channel = int(flat_idx // (height * width))
            offset = int(flat_idx % (height * width))
            y, x = divmod(offset, width)
            current_values = x_rgb[0, :, y, x].detach().cpu().tolist()
            current_values[channel] = float(desired_values[channel, y, x].item())
            candidates.append(
                SparseCandidate(
                    estimated_gain=float(score),
                    y=y,
                    x=x,
                    channel=channel,
                    new_values=tuple(float(v) for v in current_values),
                )
            )
        return candidates

    raise ValueError(f"Unsupported granularity '{granularity}'.")


def apply_sparse_candidate(x_rgb: Tensor, candidate: SparseCandidate) -> Tensor:
    x_next = x_rgb.clone()
    new_values = x_next.new_tensor(candidate.new_values)
    x_next[0, :, candidate.y, candidate.x] = new_values
    return x_next.clamp(0.0, 1.0)


def apply_sparse_candidates_batch(
    x_rgb: Tensor,
    candidates: Sequence[SparseCandidate],
) -> Tensor:
    x_batch = x_rgb.expand(len(candidates), -1, -1, -1).clone()
    for idx, candidate in enumerate(candidates):
        new_values = x_batch.new_tensor(candidate.new_values)
        x_batch[idx, :, candidate.y, candidate.x] = new_values
    return x_batch.clamp(0.0, 1.0)


def score_candidates_with_forward(
    model: torch.nn.Module,
    x_rgb: Tensor,
    candidates: Sequence[SparseCandidate],
    top1_class: int,
    top2_class: int,
    candidate_batch_size: int,
    channels_last: bool,
) -> Tuple[SparseCandidate, Tensor]:
    if not candidates:
        raise ValueError("Cannot score an empty candidate list.")

    margins: List[Tensor] = []
    batch_size = max(1, candidate_batch_size)
    with torch.inference_mode():
        for start in range(0, len(candidates), batch_size):
            chunk = candidates[start:start + batch_size]
            x_batch = apply_sparse_candidates_batch(x_rgb, chunk)
            outputs = model_forward_bcos(model, x_batch, channels_last=channels_last)
            margins.append(margin_for_classes(outputs, top1_class, top2_class).detach())
    all_margins = torch.cat(margins, dim=0)
    best_idx = int(all_margins.argmax().item())
    return candidates[best_idx], all_margins


def mark_changed(changed_mask: Tensor, candidate: SparseCandidate) -> None:
    if changed_mask.dim() == 2:
        changed_mask[candidate.y, candidate.x] = True
    else:
        if candidate.channel < 0:
            changed_mask[:, candidate.y, candidate.x] = True
        else:
            changed_mask[candidate.channel, candidate.y, candidate.x] = True


def count_changed_pixels(steps: Sequence[SparseStep]) -> int:
    return len({(step.y, step.x) for step in steps})


def run_sparse_attack(
    model: torch.nn.Module,
    x_rgb: Tensor,
    max_pixels: int,
    candidate_topk: int,
    candidate_batch_size: int,
    granularity: str,
    value_mode: str,
    epsilon: float,
    require_improvement: bool,
    min_gain: float,
    channels_last: bool,
    verbose: bool,
) -> SparseAttackResult:
    if x_rgb.shape[0] != 1:
        raise ValueError("run_sparse_attack expects a single image batch.")
    if max_pixels <= 0:
        raise ValueError("--max-pixels must be > 0.")

    x_adv = x_rgb.clone()
    _, _, height, width = x_adv.shape
    changed_mask = torch.zeros(
        (height, width) if granularity == "pixel" else (3, height, width),
        dtype=torch.bool,
        device=x_adv.device,
    )

    with torch.inference_mode():
        orig_outputs = model_forward_bcos(model, x_adv, channels_last=channels_last)
        top1_class = int(orig_outputs.argmax(dim=1).item())
        top2_class = resolve_runner_up_class(orig_outputs, top1_class)
        original_top1_logit = float(orig_outputs[0, top1_class].item())
        original_top2_logit = float(orig_outputs[0, top2_class].item())
        original_margin = float((orig_outputs[0, top2_class] - orig_outputs[0, top1_class]).item())
        original_top1_prob = probability_value(model, orig_outputs, top1_class)
        original_top2_prob = probability_value(model, orig_outputs, top2_class)

    current_margin = original_margin
    steps: List[SparseStep] = []

    for step_idx in range(1, max_pixels + 1):
        attr_top1 = extract_attribution(model, to_bcos_input(x_adv), target_class=top1_class)
        attr_top2 = extract_attribution(model, to_bcos_input(x_adv), target_class=top2_class)
        candidates = build_sparse_candidates(
            x_rgb=x_adv,
            attr_top1=attr_top1,
            attr_top2=attr_top2,
            changed_mask=changed_mask,
            candidate_topk=candidate_topk,
            granularity=granularity,
            value_mode=value_mode,
            epsilon=epsilon,
        )
        if not candidates:
            if verbose:
                print("  No sparse candidates left.")
            break
        if candidates[0].estimated_gain < min_gain:
            if verbose:
                print(
                    f"  Stop at step {step_idx}: best estimated gain "
                    f"{candidates[0].estimated_gain:.6f} < min_gain {min_gain:.6f}."
                )
            break

        margin_before = current_margin
        best_candidate, candidate_margins = score_candidates_with_forward(
            model=model,
            x_rgb=x_adv,
            candidates=candidates,
            top1_class=top1_class,
            top2_class=top2_class,
            candidate_batch_size=candidate_batch_size,
            channels_last=channels_last,
        )
        best_margin = float(candidate_margins.max().item())
        if require_improvement and best_margin <= current_margin:
            if verbose:
                print(
                    f"  Stop at step {step_idx}: verified margin would not improve "
                    f"({current_margin:.6f} -> {best_margin:.6f})."
                )
            break

        old_values = tuple(float(v) for v in x_adv[0, :, best_candidate.y, best_candidate.x].detach().cpu().tolist())
        x_adv = apply_sparse_candidate(x_adv, best_candidate)
        mark_changed(changed_mask, best_candidate)

        with torch.inference_mode():
            outputs = model_forward_bcos(model, x_adv, channels_last=channels_last)
            pred_after = int(outputs.argmax(dim=1).item())
            top1_logit_after = float(outputs[0, top1_class].item())
            top2_logit_after = float(outputs[0, top2_class].item())
            current_margin = float((outputs[0, top2_class] - outputs[0, top1_class]).item())

        sparse_step = SparseStep(
            step=step_idx,
            y=best_candidate.y,
            x=best_candidate.x,
            channel=best_candidate.channel,
            old_values=old_values,
            new_values=best_candidate.new_values,
            estimated_gain=best_candidate.estimated_gain,
            margin_before=margin_before,
            margin_after=current_margin,
            pred_after=pred_after,
            top1_logit_after=top1_logit_after,
            top2_logit_after=top2_logit_after,
        )
        steps.append(sparse_step)

        if verbose:
            channel_name = "rgb" if best_candidate.channel < 0 else str(best_candidate.channel)
            print(
                f"  Step {step_idx:03d}/{max_pixels:03d} | "
                f"pixel=({best_candidate.y},{best_candidate.x}) channel={channel_name} | "
                f"margin={current_margin:.4f} | pred={pred_after} | "
                f"est_gain={best_candidate.estimated_gain:.4f}"
            )

        if pred_after != top1_class:
            break

    with torch.inference_mode():
        final_outputs = model_forward_bcos(model, x_adv, channels_last=channels_last)
        final_class = int(final_outputs.argmax(dim=1).item())
        final_top1_logit = float(final_outputs[0, top1_class].item())
        final_top2_logit = float(final_outputs[0, top2_class].item())
        final_margin = float((final_outputs[0, top2_class] - final_outputs[0, top1_class]).item())
        final_top1_prob = probability_value(model, final_outputs, top1_class)
        final_top2_prob = probability_value(model, final_outputs, top2_class)

    return SparseAttackResult(
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


def build_image_output_dir(output_root: Path, image_path: str, image_index: Optional[int]) -> Path:
    image_file = Path(image_path)
    parts: List[str] = []
    if image_index is not None:
        parts.append(f"{image_index:03d}")
    if image_file.parent.name:
        parts.append(image_file.parent.name)
    parts.append(image_file.stem)
    safe_name = "_".join(part for part in parts if part)
    return output_root / f"{safe_name}_sparse"


def load_image_paths_from_csv(csv_path: Path) -> List[Tuple[Optional[int], str]]:
    items: List[Tuple[Optional[int], str]] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if "image_path" not in (reader.fieldnames or []):
            raise ValueError(f"CSV {csv_path} must contain an 'image_path' column.")
        for row_num, row in enumerate(reader, start=1):
            image_path = (row.get("image_path") or "").strip()
            if not image_path:
                continue
            raw_index = (row.get("index") or "").strip()
            image_index = int(raw_index) if raw_index.isdigit() else row_num
            items.append((image_index, image_path))
    if not items:
        raise ValueError(f"CSV {csv_path} does not contain any valid image_path values.")
    return items


def save_steps_csv(steps: Sequence[SparseStep], path: Path) -> None:
    fieldnames = [
        "step",
        "y",
        "x",
        "channel",
        "old_r",
        "old_g",
        "old_b",
        "new_r",
        "new_g",
        "new_b",
        "estimated_gain",
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
                    "channel": step.channel,
                    "old_r": step.old_values[0],
                    "old_g": step.old_values[1],
                    "old_b": step.old_values[2],
                    "new_r": step.new_values[0],
                    "new_g": step.new_values[1],
                    "new_b": step.new_values[2],
                    "estimated_gain": step.estimated_gain,
                    "margin_before": step.margin_before,
                    "margin_after": step.margin_after,
                    "pred_after": step.pred_after,
                    "top1_logit_after": step.top1_logit_after,
                    "top2_logit_after": step.top2_logit_after,
                }
            )


def save_changed_overlay(x_orig: Tensor, result: SparseAttackResult, path: Path) -> None:
    from PIL import Image

    image = x_orig[0].detach().cpu().permute(1, 2, 0).numpy()
    image = np.clip(image, 0.0, 1.0)
    overlay = image.copy()
    for step in result.steps:
        overlay[step.y, step.x] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    blended = np.clip(0.65 * image + 0.35 * overlay, 0.0, 1.0)
    out = Image.fromarray((blended * 255.0).round().astype(np.uint8))
    out.save(path)


def process_single_image(
    model: torch.nn.Module,
    image_path: str,
    image_index: Optional[int],
    output_root: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    x_rgb, resolved_image_path = load_rgb_image(image_path, device=args.device_obj)
    x_rgb = maybe_channels_last(x_rgb, args.channels_last)
    out_dir = build_image_output_dir(output_root, resolved_image_path, image_index)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nImage: {resolved_image_path}")
    result = run_sparse_attack(
        model=model,
        x_rgb=x_rgb,
        max_pixels=args.max_pixels,
        candidate_topk=args.candidate_topk,
        candidate_batch_size=args.candidate_batch_size,
        granularity=args.granularity,
        value_mode=args.value_mode,
        epsilon=args.epsilon,
        require_improvement=not args.allow_non_improving,
        min_gain=args.min_gain,
        channels_last=args.channels_last,
        verbose=args.verbose,
    )

    if args.save_images:
        import torchvision.utils as vutils

        vutils.save_image(x_rgb, str(out_dir / "before.png"))
        vutils.save_image(result.x_adv, str(out_dir / "after.png"))
        save_changed_overlay(x_rgb, result, out_dir / "changed_pixels.png")
    save_steps_csv(result.steps, out_dir / "steps.csv")

    print(
        "  "
        f"top1={result.original_class} runner_up={result.runner_up_class} "
        f"final={result.final_class} success={int(result.success)}"
    )
    print(
        "  "
        f"margin {result.original_margin:.4f} -> {result.final_margin:.4f} | "
        f"changed_pixels={count_changed_pixels(result.steps)} steps={len(result.steps)}"
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
        "changed_pixels": count_changed_pixels(result.steps),
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
        "granularity": args.granularity,
        "value_mode": args.value_mode,
        "epsilon": f"{args.epsilon:g}",
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
        "granularity",
        "value_mode",
        "epsilon",
    ]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedy sparse attack for B-cos models using model.explain().",
    )
    parser.add_argument("--image", type=str, default=None, help="Path to one input image")
    parser.add_argument("--images-csv", type=str, default=None, help="CSV containing an image_path column")
    parser.add_argument("--model", type=str, default="resnet50", help="B-cos model name")
    parser.add_argument("--max-pixels", type=int, default=20, help="Maximum sparse edit steps")
    parser.add_argument(
        "--candidate-topk",
        type=int,
        default=32,
        help="Number of top linearized candidates to verify with real forwards each step",
    )
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=64,
        help="Candidate images per forward pass during top-k verification",
    )
    parser.add_argument(
        "--granularity",
        choices=("pixel", "channel"),
        default="pixel",
        help="Edit all RGB channels of a pixel or one RGB channel at a time",
    )
    parser.add_argument(
        "--value-mode",
        choices=("corners", "linf"),
        default="corners",
        help="Set chosen values to 0/1 corners or move by per-step linf epsilon",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.3,
        help="Per-step RGB change used only when --value-mode linf",
    )
    parser.add_argument(
        "--min-gain",
        type=float,
        default=-float("inf"),
        help="Stop if the best estimated gain is below this value",
    )
    parser.add_argument(
        "--allow-non-improving",
        action="store_true",
        help="Apply the best verified candidate even when the fixed top2-top1 margin does not improve",
    )
    parser.add_argument("--save-images", action="store_true", help="Save before/after PNGs and changed-pixel overlay")
    parser.add_argument("--verbose", action="store_true", help="Print one line per sparse edit")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULT_DIR / "sparse_attack_outputs"),
        help="Root folder for generated outputs",
    )
    args = parser.parse_args()

    if args.image and args.images_csv:
        parser.error("Use either --image or --images-csv, not both.")
    if not args.image and not args.images_csv:
        parser.error("Provide --image or --images-csv.")
    if args.max_pixels <= 0:
        parser.error("--max-pixels must be > 0.")
    if args.candidate_topk <= 0:
        parser.error("--candidate-topk must be > 0.")
    if args.candidate_batch_size <= 0:
        parser.error("--candidate-batch-size must be > 0.")
    if args.value_mode == "linf" and args.epsilon <= 0:
        parser.error("--epsilon must be > 0 when --value-mode linf.")
    return args


def main() -> None:
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    configure_fast_runtime(device)
    args.device_obj = device
    args.channels_last = device.type == "cuda"

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  B-cos Sparse Explain Attack")
    print("=" * 72)
    print(f"  Device      : {device}")
    print(f"  Model       : {args.model}")
    print(f"  Max pixels  : {args.max_pixels}")
    print(f"  Top-k verify: {args.candidate_topk}")
    print(f"  Granularity : {args.granularity}")
    print(f"  Value mode  : {args.value_mode}")
    print(f"  Output root : {output_root}")

    print("\nLoading B-cos model ...")
    model_fn = getattr(bcos.pretrained, args.model)
    model = model_fn(pretrained=True).to(device).eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.images_csv:
        image_items = load_image_paths_from_csv(Path(args.images_csv))
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
