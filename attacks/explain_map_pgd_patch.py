"""
Explain-map driven iterative patch attack for B-cos models.

This script does not optimize with ES and does not read gradients directly.
Instead, each update step uses only the outputs of `model.explain()`:
* `contribution_map` to locate and weight target-supporting regions,
* `explanation` RGBA to derive a pseudo update direction.

The update behaves like patch-PGD:
* iterative,
* projected to an l0/l2/linf patch budget,
* optional momentum,
* optional blur after each step for smoother patches.
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from attack_utils import canonicalize_norm, project_perturbation

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
from bcos_es_patch import (
    apply_patch,
    extract_attribution,
    load_rgb_image,
    to_bcos_input,
    visualize_patch_results,
)


def box_blur(x: Tensor, kernel_size: int) -> Tensor:
    h, w = x.shape[-2:]
    max_kernel = min(h, w)
    if max_kernel <= 1 or kernel_size <= 1:
        return x
    if max_kernel % 2 == 0:
        max_kernel -= 1
    kernel_size = min(kernel_size, max_kernel)
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        kernel_size -= 1
    if kernel_size <= 1:
        return x
    pad = kernel_size // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(x_pad, kernel_size, stride=1)


def to_rgba_tensor(explanation: np.ndarray, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.from_numpy(explanation).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)


def squeeze_contribution_map(contribution_map: Tensor) -> Tensor:
    cmap = contribution_map
    if cmap.dim() == 4:
        cmap = cmap.squeeze(0)
    if cmap.dim() == 3:
        cmap = cmap.squeeze(0)
    return cmap


def normalize_map(x: Tensor) -> Tensor:
    x = x - x.amin(dim=(-2, -1), keepdim=True)
    return x / (x.amax(dim=(-2, -1), keepdim=True) + 1e-6)


def resolve_plot_save_path(save_arg: str, out_dir: Path) -> str:
    if save_arg == "explain_map_pgd_patch.png":
        return str(out_dir / "explain_map_pgd_patch.png")
    save_path = Path(save_arg)
    if save_path.is_absolute():
        return str(save_path)
    return str(out_dir / save_path.name)


def find_best_patch_position(
    contribution_map: Tensor,
    explanation_rgba: Tensor,
    patch_size: int,
) -> Tuple[int, int]:
    cmap = squeeze_contribution_map(contribution_map)
    alpha = explanation_rgba[0, 3]
    score = torch.relu(cmap) * (0.5 + 0.5 * alpha)

    attr = score.detach().cpu().numpy().astype(np.float64)
    h, w = attr.shape
    s = min(patch_size, h, w)
    prefix = np.zeros((h + 1, w + 1), dtype=np.float64)
    prefix[1:, 1:] = np.cumsum(np.cumsum(attr, axis=0), axis=1)

    best_sum = -1.0
    best_y, best_x = 0, 0
    for y in range(h - s + 1):
        for x in range(w - s + 1):
            window_sum = (
                prefix[y + s, x + s]
                - prefix[y, x + s]
                - prefix[y + s, x]
                + prefix[y, x]
            )
            if window_sum > best_sum:
                best_sum = window_sum
                best_y, best_x = y, x
    return best_y, best_x


def build_step_direction(
    x_rgb: Tensor,
    explanation_rgba: Tensor,
    contribution_map: Tensor,
    pos_y: int,
    pos_x: int,
    patch_size: int,
    alpha_blur: int,
    mask_floor: float,
) -> Tuple[Tensor, Tensor]:
    crop = x_rgb[:, :, pos_y:pos_y + patch_size, pos_x:pos_x + patch_size]
    expl_crop = explanation_rgba[:, :, pos_y:pos_y + patch_size, pos_x:pos_x + patch_size]
    expl_rgb = expl_crop[:, :3]
    alpha = expl_crop[:, 3:4]

    cmap = squeeze_contribution_map(contribution_map)
    cmap_crop = cmap[pos_y:pos_y + patch_size, pos_x:pos_x + patch_size].unsqueeze(0).unsqueeze(0)
    pos_support = normalize_map(torch.relu(cmap_crop))
    alpha = normalize_map(alpha)
    alpha = box_blur(alpha, alpha_blur)

    weight = (mask_floor + (1.0 - mask_floor) * alpha) * (0.35 + 0.65 * pos_support)
    anti_expl = (0.5 - expl_rgb) * 2.0
    wash_to_mean = crop.mean(dim=(2, 3), keepdim=True) - crop
    blur_pull = box_blur(crop, 7) - crop

    direction = (0.60 * anti_expl + 0.25 * wash_to_mean + 0.15 * blur_pull) * weight
    direction = box_blur(direction, 5)
    return direction, weight


def compute_patch_score(
    model: torch.nn.Module,
    x_rgb: Tensor,
    patch: Tensor,
    pos_y: int,
    pos_x: int,
    target_class: int,
) -> Tuple[float, float, int, float]:
    x_adv = apply_patch(x_rgb, patch, pos_y, pos_x)
    with torch.no_grad():
        outputs = model(to_bcos_input(x_adv))
        pred_class = outputs.argmax(1).item()
        target_logit = outputs[0, target_class].item()
        pred_logit = outputs[0, pred_class].item()
        other = outputs.clone()
        other[:, target_class] = -float("inf")
        score = (other.max(dim=1)[0] - outputs[:, target_class])[0].item()
    return score, target_logit, pred_class, pred_logit


def run_explain_map_attack(
    model: torch.nn.Module,
    x_rgb: Tensor,
    target_class: int,
    patch_size: int,
    pos_y: int,
    pos_x: int,
    steps: int,
    step_size: float,
    epsilon: float,
    norm: str,
    momentum: float,
    blur_kernel: int,
    alpha_blur: int,
    mask_floor: float,
    use_explain_position: bool,
    verbose: bool,
) -> Tuple[Tensor, int, int, List[Tuple[int, float, float, float]]]:
    initial_attr = extract_attribution(model, to_bcos_input(x_rgb), target_class=target_class)
    initial_rgba = to_rgba_tensor(initial_attr["explanation"], x_rgb.device, x_rgb.dtype)

    if use_explain_position:
        pos_y, pos_x = find_best_patch_position(
            initial_attr["contribution_map"],
            initial_rgba,
            patch_size,
        )

    patch = torch.zeros(1, 3, patch_size, patch_size, device=x_rgb.device, dtype=x_rgb.dtype)
    velocity = torch.zeros_like(patch)
    history: List[Tuple[int, float, float, float]] = []

    for step in range(steps):
        x_adv = apply_patch(x_rgb, patch, pos_y, pos_x)
        attr = extract_attribution(model, to_bcos_input(x_adv), target_class=target_class)
        explanation_rgba = to_rgba_tensor(attr["explanation"], x_rgb.device, x_rgb.dtype)

        direction, weight = build_step_direction(
            x_adv,
            explanation_rgba,
            attr["contribution_map"],
            pos_y,
            pos_x,
            patch_size,
            alpha_blur=alpha_blur,
            mask_floor=mask_floor,
        )

        step_dir = direction.sign()
        velocity = momentum * velocity + (1.0 - momentum) * step_dir
        patch = patch + step_size * velocity.sign()
        patch = patch * weight
        patch = box_blur(patch, blur_kernel)
        patch = project_perturbation(patch, epsilon, norm)

        score, target_logit, pred_class, pred_logit = compute_patch_score(
            model,
            x_rgb,
            patch,
            pos_y,
            pos_x,
            target_class,
        )
        history.append((step, score, target_logit, 0.0))
        if verbose:
            print(
                f"  Step {step + 1:3d}/{steps} | "
                f"Target Logit: {target_logit:7.3f} | "
                f"Score: {score:7.3f} | Pred: {pred_class:4d}"
            )
            _ = pred_logit

    return patch, pos_y, pos_x, history


def main() -> None:
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser(
        description="Explain-map iterative patch attack for B-cos models",
    )
    parser.add_argument("--image", type=str, required=True, help="Path to one input image")
    parser.add_argument("--model", type=str, default="resnet18", help="B-cos model name")
    parser.add_argument("--patch-size", type=int, default=32, help="Square patch size")
    parser.add_argument("--pos-y", type=int, default=-1, help="Manual patch Y position")
    parser.add_argument("--pos-x", type=int, default=-1, help="Manual patch X position")
    parser.add_argument("--steps", type=int, default=80, help="Number of iterative updates")
    parser.add_argument("--step-size", type=float, default=0.025, help="Per-step patch update size")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.30,
        help="Perturbation budget. For l0, values <= 1 are treated as a fraction of patch coefficients.",
    )
    parser.add_argument("--norm", type=str, default="linf", help="Perturbation norm budget: l0, l2, or linf")
    parser.add_argument("--momentum", type=float, default=0.8, help="Momentum on pseudo direction")
    parser.add_argument("--blur-kernel", type=int, default=5, help="Blur applied after each patch update")
    parser.add_argument("--alpha-blur", type=int, default=9, help="Blur on explain alpha/support mask")
    parser.add_argument("--mask-floor", type=float, default=0.08, help="Minimum explain mask strength")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULT_DIR / "explain_map_pgd_outputs"),
        help="Root folder for all generated outputs",
    )
    parser.add_argument("--save", type=str, default="explain_map_pgd_patch.png", help="Output plot path")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 72)
    print("  Explain-map Iterative Patch Attack")
    print("=" * 72)
    print(f"  Device       : {device}")
    print(f"  Model        : {args.model}")
    print(f"  Patch size   : {args.patch_size}x{args.patch_size}")
    print(f"  Budget       : {canonicalize_norm(args.norm)} <= {args.epsilon:g}")

    print("\n▸ Loading B-cos model ...")
    model_fn = getattr(bcos.pretrained, args.model)
    model = model_fn(pretrained=True).to(device).eval()

    print("\n▸ Loading image ...")
    x_rgb, image_path = load_rgb_image(args.image, device=device)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_root = Path(args.output_dir).resolve()
    out_dir = output_root / f"{image_name}_explain_map_iter"
    out_dir.mkdir(parents=True, exist_ok=True)

    import torchvision.utils as vutils

    before_path = out_dir / "before.png"
    after_path = out_dir / "after.png"
    vutils.save_image(x_rgb, str(before_path))
    print(f"  Saved original image to {before_path}")

    with torch.no_grad():
        out_orig = model(to_bcos_input(x_rgb))
        target_class = out_orig.argmax(1).item()
        logit_orig = out_orig[0, target_class].item()

    print(f"  Original class: {target_class} (logit: {logit_orig:.3f})")

    use_explain_position = args.pos_y < 0 or args.pos_x < 0
    pos_y = max(0, args.pos_y)
    pos_x = max(0, args.pos_x)

    print("\n▸ Running explain-map iterative attack ...")
    final_patch, final_pos_y, final_pos_x, history = run_explain_map_attack(
        model=model,
        x_rgb=x_rgb,
        target_class=target_class,
        patch_size=args.patch_size,
        pos_y=pos_y,
        pos_x=pos_x,
        steps=args.steps,
        step_size=args.step_size,
        epsilon=args.epsilon,
        norm=args.norm,
        momentum=args.momentum,
        blur_kernel=args.blur_kernel,
        alpha_blur=args.alpha_blur,
        mask_floor=args.mask_floor,
        use_explain_position=use_explain_position,
        verbose=True,
    )
    print(f"  Final patch position: ({final_pos_y}, {final_pos_x})")

    final_patch = project_perturbation(final_patch, args.epsilon, args.norm)
    x_pert_rgb = apply_patch(x_rgb, final_patch, final_pos_y, final_pos_x)
    vutils.save_image(x_pert_rgb, str(after_path))
    print(f"  Saved perturbed image to {after_path}")

    with torch.no_grad():
        out_pert = model(to_bcos_input(x_pert_rgb))
        pred_class_pert = out_pert.argmax(1).item()
        logit_pert_tgt = out_pert[0, target_class].item()
        logit_pert_pred = out_pert[0, pred_class_pert].item()

    print("\n▸ Attack complete")
    print(f"  Target class logit: {logit_orig:.3f} -> {logit_pert_tgt:.3f}")
    if pred_class_pert != target_class:
        print(f"  Class changed to {pred_class_pert} (logit: {logit_pert_pred:.3f})")
    else:
        print("  Prediction unchanged.")

    attr_orig = extract_attribution(model, to_bcos_input(x_rgb), target_class=target_class)
    attr_pert = extract_attribution(model, to_bcos_input(x_pert_rgb), target_class=target_class)

    save_plot_path = resolve_plot_save_path(args.save, out_dir)
    visualize_patch_results(
        x_orig_rgb=x_rgb,
        x_pert_rgb=x_pert_rgb,
        patch=final_patch,
        attr_orig=attr_orig,
        attr_pert=attr_pert,
        target_class=target_class,
        logit_orig=logit_orig,
        pred_class_pert=pred_class_pert,
        logit_pert_tgt=logit_pert_tgt,
        logit_pert_pred=logit_pert_pred,
        history=history,
        save_path=save_plot_path,
    )

    print("\n" + "=" * 72)
    print("  Done")
    print("=" * 72)


if __name__ == "__main__":
    main()
