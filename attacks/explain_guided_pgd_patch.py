"""
Explain-guided iterative patch attack for B-cos models.

This variant keeps the explain-driven patch placement from
`explain_guided_pixel_es_patch.py`: choose the fixed patch location once by
maximizing the summed absolute contribution inside the patch window.

After that, it updates the perturbation using the B-cos
`dynamic_linear_weights` from `model.explain()`, projected to an
l0/l2/linf budget.
"""

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import models as tv_models

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
from explain_guided_pixel_es_patch import (
    build_image_output_dir,
    configure_fast_runtime,
    load_image_paths_from_csv,
    maybe_channels_last,
    model_forward_rgb,
    resolve_amp_dtype,
    resolve_plot_save_path,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TORCHVISION_MODEL_MAP = {
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "resnet50": "resnet50",
    "resnet50_long": "resnet50",
    "resnet101": "resnet101",
    "resnet152": "resnet152",
    "resnext50_32x4d": "resnext50_32x4d",
    "densenet121": "densenet121",
    "densenet161": "densenet161",
    "densenet169": "densenet169",
    "densenet201": "densenet201",
    "convnext_tiny": "convnext_tiny",
    "convnext_base": "convnext_base",
    "vgg11_bnu": "vgg11_bn",
}


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


def normalize_for_original_model(x_rgb: Tensor) -> Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x_rgb.device, dtype=x_rgb.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x_rgb.device, dtype=x_rgb.dtype).view(1, 3, 1, 1)
    return (x_rgb - mean) / std


def model_forward_original(
    model: torch.nn.Module,
    x_rgb: Tensor,
    amp_dtype: Optional[torch.dtype],
    channels_last: bool,
) -> Tensor:
    x_norm = normalize_for_original_model(x_rgb)
    x_norm = maybe_channels_last(x_norm, channels_last)
    if amp_dtype is None or x_rgb.device.type != "cuda":
        return model(x_norm)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        return model(x_norm)


def model_forward_attack(
    model: torch.nn.Module,
    x_rgb: Tensor,
    amp_dtype: Optional[torch.dtype],
    channels_last: bool,
    attack_original_model: bool,
) -> Tensor:
    if attack_original_model:
        return model_forward_original(
            model,
            x_rgb,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
        )
    return model_forward_rgb(
        model,
        x_rgb,
        amp_dtype=amp_dtype,
        channels_last=channels_last,
    )


def resolve_torchvision_model_name(model_name: str) -> str:
    key = model_name.strip()
    if key in TORCHVISION_MODEL_MAP:
        return TORCHVISION_MODEL_MAP[key]
    raise ValueError(
        f"No torchvision equivalent configured for '{model_name}'. "
        f"Supported: {', '.join(sorted(TORCHVISION_MODEL_MAP))}"
    )


def resolve_original_model_weights(model_name: str):
    weights_enum = tv_models.get_model_weights(model_name)
    weight_v1 = getattr(weights_enum, "IMAGENET1K_V1", None)
    if weight_v1 is not None:
        return weight_v1
    return weights_enum.DEFAULT


def load_original_attack_model(
    model_name: str,
    device: torch.device,
    channels_last: bool,
) -> Tuple[torch.nn.Module, str]:
    torchvision_name = resolve_torchvision_model_name(model_name)
    weights = resolve_original_model_weights(torchvision_name)
    model = tv_models.get_model(torchvision_name, weights=weights).to(device).eval()
    if channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model, torchvision_name


def compute_cw_score(outputs: Tensor, target_class: int) -> Tensor:
    other = outputs.clone()
    other[:, target_class] = -float("inf")
    return other.max(dim=1)[0] - outputs[:, target_class]


def find_best_patch_position(contribution_map: Tensor, patch_size: int) -> Tuple[int, int]:
    cmap = squeeze_contribution_map(contribution_map)
    attr = cmap.abs().detach().cpu().numpy().astype(np.float64)
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


def dynamic_linear_weights_to_rgb_gradient(dynamic_linear_weights: Tensor) -> Tensor:
    if dynamic_linear_weights.dim() == 3:
        dynamic_linear_weights = dynamic_linear_weights.unsqueeze(0)
    if dynamic_linear_weights.dim() != 4 or dynamic_linear_weights.shape[1] != 6:
        raise ValueError(
            "Expected dynamic_linear_weights with shape [B, 6, H, W] from model.explain()."
        )

    # B-cos consumes [x, 1 - x]. Convert the 6-channel gradient back to RGB space.
    return dynamic_linear_weights[:, :3] - dynamic_linear_weights[:, 3:]


def build_step_direction(
    dynamic_linear_weights: Tensor,
    contribution_map: Tensor,
    pos_y: int,
    pos_x: int,
    patch_size: int,
    alpha_blur: int,
    mask_floor: float,
    use_support_mask: bool,
    use_support_blur: bool,
) -> Tuple[Tensor, Tensor]:
    grad_rgb = dynamic_linear_weights_to_rgb_gradient(dynamic_linear_weights)
    grad_crop = grad_rgb[:, :, pos_y:pos_y + patch_size, pos_x:pos_x + patch_size]
    if use_support_mask:
        cmap = squeeze_contribution_map(contribution_map)
        cmap_crop = cmap[pos_y:pos_y + patch_size, pos_x:pos_x + patch_size].unsqueeze(0).unsqueeze(0)
        pos_support = normalize_map(torch.relu(cmap_crop))
        grad_support = normalize_map(grad_crop.norm(p=2, dim=1, keepdim=True))
        if use_support_blur:
            grad_support = box_blur(grad_support, alpha_blur)
        weight = (mask_floor + (1.0 - mask_floor) * grad_support) * (0.35 + 0.65 * pos_support)
    else:
        weight = torch.ones_like(grad_crop[:, :1])
    direction = -grad_crop
    return direction, weight


def compute_patch_score(
    model: torch.nn.Module,
    x_rgb: Tensor,
    patch: Tensor,
    pos_y: int,
    pos_x: int,
    target_class: int,
    amp_dtype: Optional[torch.dtype],
    channels_last: bool,
) -> Tuple[float, float, int, float]:
    x_adv = apply_patch(x_rgb, patch, pos_y, pos_x)
    with torch.inference_mode():
        outputs = model_forward_attack(
            model,
            x_adv,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
            attack_original_model=False,
        )
        pred_class = outputs.argmax(1).item()
        target_logit = outputs[0, target_class].item()
        pred_logit = outputs[0, pred_class].item()
        score = compute_cw_score(outputs, target_class)[0].item()
    return score, target_logit, pred_class, pred_logit


def compute_patch_score_original(
    model: torch.nn.Module,
    x_rgb: Tensor,
    patch: Tensor,
    pos_y: int,
    pos_x: int,
    target_class: int,
    amp_dtype: Optional[torch.dtype],
    channels_last: bool,
) -> Tuple[float, float, int, float]:
    x_adv = apply_patch(x_rgb, patch, pos_y, pos_x)
    with torch.inference_mode():
        outputs = model_forward_attack(
            model,
            x_adv,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
            attack_original_model=True,
        )
        pred_class = outputs.argmax(1).item()
        target_logit = outputs[0, target_class].item()
        pred_logit = outputs[0, pred_class].item()
        score = compute_cw_score(outputs, target_class)[0].item()
    return score, target_logit, pred_class, pred_logit


def run_explain_guided_pgd_attack(
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
    disable_mask: bool,
    disable_blur: bool,
    amp_dtype: Optional[torch.dtype],
    channels_last: bool,
    verbose: bool,
) -> Tuple[Tensor, int, int, List[Tuple[int, float, float, float]]]:
    initial_attr = extract_attribution(model, to_bcos_input(x_rgb), target_class=target_class)
    if use_explain_position:
        pos_y, pos_x = find_best_patch_position(
            initial_attr["contribution_map"],
            patch_size,
        )

    patch = torch.zeros(1, 3, patch_size, patch_size, device=x_rgb.device, dtype=x_rgb.dtype)
    velocity = torch.zeros_like(patch)
    history: List[Tuple[int, float, float, float]] = []
    best_patch = patch.clone()
    best_score = -float("inf")
    full_image_patch = patch_size >= x_rgb.shape[-2] and patch_size >= x_rgb.shape[-1]
    use_support_mask = not disable_mask and not full_image_patch
    use_support_blur = not disable_blur
    patch_blur_kernel = 1 if disable_blur else blur_kernel

    for step in range(steps):
        x_adv = apply_patch(x_rgb, patch, pos_y, pos_x)
        attr = extract_attribution(model, to_bcos_input(x_adv), target_class=target_class)

        direction, weight = build_step_direction(
            attr["dynamic_linear_weights"],
            attr["contribution_map"],
            pos_y,
            pos_x,
            patch_size,
            alpha_blur=alpha_blur,
            mask_floor=mask_floor,
            use_support_mask=use_support_mask,
            use_support_blur=use_support_blur,
        )

        step_dir = direction.sign()
        velocity = momentum * velocity + (1.0 - momentum) * step_dir
        patch = patch + step_size * weight * velocity.sign()
        patch = box_blur(patch, patch_blur_kernel)
        patch = project_perturbation(patch, epsilon, norm)

        score, target_logit, pred_class, pred_logit = compute_patch_score(
            model,
            x_rgb,
            patch,
            pos_y,
            pos_x,
            target_class,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
        )
        if score > best_score:
            best_score = score
            best_patch = patch.clone()
        history.append((step, score, target_logit, 0.0))
        if verbose:
            print(
                f"  Step {step + 1:3d}/{steps} | "
                f"Target Logit: {target_logit:7.3f} | "
                f"Score: {score:7.3f} | Pred: {pred_class:4d}"
            )
            _ = pred_logit

    return best_patch, pos_y, pos_x, history


def run_original_model_pgd_attack(
    guide_model: torch.nn.Module,
    attack_model: torch.nn.Module,
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
    use_explain_position: bool,
    amp_dtype: Optional[torch.dtype],
    channels_last: bool,
    verbose: bool,
) -> Tuple[Tensor, int, int, List[Tuple[int, float, float, float]]]:
    initial_attr = extract_attribution(guide_model, to_bcos_input(x_rgb), target_class=target_class)
    if use_explain_position:
        pos_y, pos_x = find_best_patch_position(
            initial_attr["contribution_map"],
            patch_size,
        )

    patch = torch.zeros(1, 3, patch_size, patch_size, device=x_rgb.device, dtype=x_rgb.dtype)
    velocity = torch.zeros_like(patch)
    history: List[Tuple[int, float, float, float]] = []
    best_patch = patch.clone()
    best_score = -float("inf")

    for step in range(steps):
        patch_var = patch.detach().clone().requires_grad_(True)
        x_adv = apply_patch(x_rgb, patch_var, pos_y, pos_x)
        outputs = model_forward_attack(
            attack_model,
            x_adv,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
            attack_original_model=True,
        )
        score = compute_cw_score(outputs, target_class)[0]
        grad = torch.autograd.grad(score, patch_var)[0]

        grad_sign = grad.sign()
        velocity = momentum * velocity + (1.0 - momentum) * grad_sign
        patch = patch + step_size * velocity.sign()
        patch = project_perturbation(patch, epsilon, norm)

        score_value, target_logit, pred_class, pred_logit = compute_patch_score_original(
            attack_model,
            x_rgb,
            patch,
            pos_y,
            pos_x,
            target_class,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
        )
        if score_value > best_score:
            best_score = score_value
            best_patch = patch.clone()
        history.append((step, score_value, target_logit, 0.0))
        if verbose:
            print(
                f"  Step {step + 1:3d}/{steps} | "
                f"Target Logit: {target_logit:7.3f} | "
                f"Score: {score_value:7.3f} | Pred: {pred_class:4d}"
            )
            _ = pred_logit

    return best_patch, pos_y, pos_x, history


def process_single_image(
    guide_model: torch.nn.Module,
    attack_model: torch.nn.Module,
    image_path: str,
    image_index: Optional[int],
    output_root: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    x_rgb, resolved_image_path = load_rgb_image(image_path, device=args.device_obj)
    x_rgb = maybe_channels_last(x_rgb, args.channels_last)
    out_dir = build_image_output_dir(output_root, resolved_image_path, image_index)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n▸ Loading image ... {resolved_image_path}")

    before_path = out_dir / "before.png"
    after_path = out_dir / "after.png"
    if args.save_images:
        import torchvision.utils as vutils

        vutils.save_image(x_rgb, str(before_path))
        print(f"  Saved original image to {before_path}")

    with torch.inference_mode():
        out_orig = model_forward_attack(
            attack_model,
            x_rgb,
            amp_dtype=args.amp_dtype,
            channels_last=args.channels_last,
            attack_original_model=args.attack_original_model,
        )
        target_class = out_orig.argmax(1).item()
        logit_orig = out_orig[0, target_class].item()

    print(f"  Original class: {target_class} (logit: {logit_orig:.3f})")

    use_explain_position = args.pos_y < 0 or args.pos_x < 0
    pos_y = max(0, args.pos_y)
    pos_x = max(0, args.pos_x)

    attack_label = "original-model PGD attack" if args.attack_original_model else "explain-guided PGD attack"
    print(f"\n▸ Running {attack_label} ...")
    if args.attack_original_model:
        final_patch, final_pos_y, final_pos_x, history = run_original_model_pgd_attack(
            guide_model=guide_model,
            attack_model=attack_model,
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
            use_explain_position=use_explain_position,
            amp_dtype=args.amp_dtype,
            channels_last=args.channels_last,
            verbose=args.verbose_steps,
        )
    else:
        final_patch, final_pos_y, final_pos_x, history = run_explain_guided_pgd_attack(
            model=guide_model,
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
            disable_mask=args.disable_mask,
            disable_blur=args.disable_blur,
            amp_dtype=args.amp_dtype,
            channels_last=args.channels_last,
            verbose=args.verbose_steps,
        )
    print(f"  Final patch position: ({final_pos_y}, {final_pos_x})")

    final_patch = project_perturbation(final_patch, args.epsilon, args.norm)
    x_pert_rgb = apply_patch(x_rgb, final_patch, final_pos_y, final_pos_x)
    if args.save_images:
        import torchvision.utils as vutils

        vutils.save_image(x_pert_rgb, str(after_path))
        print(f"  Saved perturbed image to {after_path}")

    with torch.inference_mode():
        out_pert = model_forward_attack(
            attack_model,
            x_pert_rgb,
            amp_dtype=args.amp_dtype,
            channels_last=args.channels_last,
            attack_original_model=args.attack_original_model,
        )
        pred_class_pert = out_pert.argmax(1).item()
        logit_pert_tgt = out_pert[0, target_class].item()
        logit_pert_pred = out_pert[0, pred_class_pert].item()

    print("\n▸ Attack complete")
    print(f"  Target class logit: {logit_orig:.3f} -> {logit_pert_tgt:.3f}")
    if pred_class_pert != target_class:
        print(f"  Class changed to {pred_class_pert} (logit: {logit_pert_pred:.3f})")
    else:
        print("  Prediction unchanged.")

    if args.save_figure:
        attr_orig = extract_attribution(guide_model, to_bcos_input(x_rgb), target_class=target_class)
        attr_pert = extract_attribution(guide_model, to_bcos_input(x_pert_rgb), target_class=target_class)

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
        print(f"  Saved figure to {save_plot_path}")

    return {
        "index": image_index,
        "image_path": resolved_image_path,
        "output_dir": str(out_dir),
        "target_class": target_class,
        "original_logit": logit_orig,
        "perturbed_class": pred_class_pert,
        "perturbed_target_logit": logit_pert_tgt,
        "perturbed_pred_logit": logit_pert_pred,
        "patch_pos_y": final_pos_y,
        "patch_pos_x": final_pos_x,
        "norm": canonicalize_norm(args.norm),
        "epsilon": f"{args.epsilon:g}",
        "success": int(pred_class_pert != target_class),
    }


def main() -> None:
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser(
        description="Explain-guided PGD-style patch attack for B-cos models",
    )
    parser.add_argument("--image", type=str, default=None, help="Path to one input image")
    parser.add_argument("--images-csv", type=str, default=None, help="CSV containing an image_path column")
    parser.add_argument("--model", type=str, default="resnet18", help="B-cos model name")
    parser.add_argument("--patch-size", type=int, default=40, help="Square patch size")
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
    parser.add_argument("--disable-mask", action="store_true", help="Disable explain/support masking of the update")
    parser.add_argument("--disable-blur", action="store_true", help="Disable support smoothing and patch blur")
    parser.add_argument(
        "--attack-original-model",
        action="store_true",
        help="Use B-cos only to choose the patch position, then optimize/evaluate on the corresponding torchvision model.",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="auto",
        help="Forward AMP dtype for eval: auto, none, bfloat16, float16.",
    )
    parser.add_argument("--save-images", action="store_true", help="Save before/after PNGs for each image")
    parser.add_argument("--save-figure", action="store_true", help="Render and save the explanation figure for each image")
    parser.add_argument("--verbose-steps", action="store_true", help="Print one log line per iterative step")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULT_DIR / "explain_guided_pgd_outputs"),
        help="Root folder for all generated outputs",
    )
    parser.add_argument("--save", type=str, default="explain_guided_pgd_patch.png", help="Output plot filename or path")
    args = parser.parse_args()

    if args.image and args.images_csv:
        parser.error("Use either --image or --images-csv, not both.")
    if not args.image and not args.images_csv:
        parser.error("Provide --image or --images-csv.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    configure_fast_runtime(device)
    args.device_obj = device
    args.amp_dtype = resolve_amp_dtype(args.amp_dtype, device)
    args.channels_last = device.type == "cuda"

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Explain-guided PGD Patch Attack")
    print("=" * 72)
    print(f"  Device       : {device}")
    print(f"  Model        : {args.model}")
    print(f"  Patch size   : {args.patch_size}x{args.patch_size}")
    print(f"  Budget       : {canonicalize_norm(args.norm)} <= {args.epsilon:g}")
    print(f"  Steps        : {args.steps}")
    print(f"  Step size    : {args.step_size:g}")
    print(f"  AMP dtype    : {args.amp_dtype}")
    print(f"  Attack mode  : {'torchvision-original' if args.attack_original_model else 'bcos-explain-pgd'}")
    if args.attack_original_model:
        print("  Guide only   : B-cos explain chooses position; optimize/eval run on torchvision model")
    print(f"  Disable mask : {args.disable_mask}")
    print(f"  Disable blur : {args.disable_blur}")
    print(f"  Save images  : {args.save_images}")
    print(f"  Save figure  : {args.save_figure}")
    print(f"  Output root  : {output_root}")

    print("\n▸ Loading B-cos guide model ...")
    model_fn = getattr(bcos.pretrained, args.model)
    guide_model = model_fn(pretrained=True).to(device).eval()
    if args.channels_last:
        guide_model = guide_model.to(memory_format=torch.channels_last)

    if args.attack_original_model:
        print("▸ Loading original attack model ...")
        attack_model, attack_model_name = load_original_attack_model(
            args.model,
            device=device,
            channels_last=args.channels_last,
        )
        print(f"  Using torchvision model: {attack_model_name}")
    else:
        attack_model = guide_model

    if args.images_csv:
        image_items = load_image_paths_from_csv(Path(args.images_csv))
        print(f"\n▸ Loaded {len(image_items)} image paths from {args.images_csv}")
    else:
        image_items = [(None, args.image)]

    results: List[Dict[str, Any]] = []
    total = len(image_items)
    for item_num, (image_index, image_path) in enumerate(image_items, start=1):
        print("\n" + "-" * 72)
        print(f"  Image {item_num}/{total}")
        print("-" * 72)
        results.append(process_single_image(guide_model, attack_model, image_path, image_index, output_root, args))

    if len(results) > 1:
        summary_path = output_root / "summary.csv"
        fieldnames = [
            "index",
            "image_path",
            "output_dir",
            "target_class",
            "original_logit",
            "perturbed_class",
            "perturbed_target_logit",
            "perturbed_pred_logit",
            "patch_pos_y",
            "patch_pos_x",
            "norm",
            "epsilon",
            "success",
        ]
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        successes = sum(row["success"] for row in results)
        print(f"\n▸ Summary saved to {summary_path}")
        print(f"  Successful class changes: {successes}/{len(results)}")

    print("\n" + "=" * 72)
    print("  Done")
    print("=" * 72)


if __name__ == "__main__":
    main()
