"""
White-box optimization on B-cos W1->L explanations.

All attack objectives use fast_forward_matrix_linear_weights():
1. Freeze the B-cos linearization at the current input.
2. Get the target-class W1->L row with one reverse-mode pass.
3. Build the selected objective from those weights and backpropagate to delta.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import random
import sys
from typing import Optional, Tuple

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from attack_utils import canonicalize_norm, project_perturbation


def setup_sys_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    local_bcos_dir = project_root / "B-cos-v2"
    if local_bcos_dir.is_dir():
        sys.path.insert(0, str(local_bcos_dir))
    elif Path("/kaggle/working/B-cos-v2").is_dir():
        sys.path.insert(0, "/kaggle/working/B-cos-v2")


def to_bcos_input(x_rgb: Tensor, add_inverse: "object") -> Tensor:
    """Build [B, 6, H, W] from [B, 3, H, W]."""
    return add_inverse(x_rgb)


def load_rgb_input(image_path: str, model, device: torch.device) -> Tensor:
    """
    Load RGB image using the exact transform attached to the selected B-cos model.
    `model.transform` returns 6-channel B-cos tensor.
    """
    with Image.open(image_path) as img:
        x_6ch = model.transform(img.convert("RGB")).unsqueeze(0).to(device=device)
    if x_6ch.dim() != 4:
        raise ValueError(f"Expected 4D transformed input, got {tuple(x_6ch.shape)}")
    if x_6ch.shape[1] == 3:
        raise ValueError(
            "This B-cos model should use AddInverse transform and return 6 channels."
        )
    if x_6ch.shape[1] < 3:
        raise ValueError(
            f"Transformed input must have at least 3 channels, got {x_6ch.shape[1]}"
        )
    return x_6ch[:, :3].clamp(0.0, 1.0).detach().clone()


def _load_image_as_tensor(path: str, target_hw: Tuple[int, int], rgba: bool = False) -> Tensor:
    """Load PNG/JPG/npy as [1, C, H, W] tensor."""
    p = Path(path)
    if p.suffix.lower() == ".npy":
        arr = np.load(str(p))
        arr = np.nan_to_num(arr.astype(np.float32))
        if arr.ndim == 3:
            allowed_channels = (1, 3, 4) if rgba else (1, 3, 6)
            if arr.shape[0] in allowed_channels:
                pass
            elif arr.shape[-1] in allowed_channels:
                arr = np.transpose(arr, (2, 0, 1))
            else:
                raise ValueError(f"Unsupported npy map shape: {arr.shape}")
        elif arr.ndim != 2:
            raise ValueError(f"Unsupported npy map shape: {arr.shape}")

        if arr.ndim == 2:
            arr = arr[None, ...]

        allowed_channels = (1, 3, 4) if rgba else (1, 3, 6)
        if arr.shape[0] not in allowed_channels:
            raise ValueError(f"Unsupported npy map shape: {arr.shape}")
        arr = torch.from_numpy(arr)
        if arr.shape[-2:] != target_hw:
            arr = arr[None, ...]
            arr = F.interpolate(
                arr,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )[0]
        return arr[None]
    else:
        map_hw = Image.open(str(p)).convert("RGBA" if rgba else "RGB")

    if map_hw.size != (target_hw[1], target_hw[0]):
        map_hw = map_hw.resize((target_hw[1], target_hw[0]))
    arr = np.array(map_hw).astype(np.float32) / 255.0
    arr = np.nan_to_num(arr)
    if arr.ndim == 3:
        arr = np.transpose(arr, (2, 0, 1))
    elif arr.ndim == 2:
        arr = arr[None, ...]
    else:
        raise ValueError(f"Unsupported map image shape: {arr.shape}")
    arr = torch.from_numpy(arr)
    return arr[None]


def normalize_map_like(template: Tensor, map_t: Tensor, channels: int = 1) -> Tensor:
    """Return map in shape [B, channels, H, W], with batch 1 and target HW."""
    b, c, h, w = template.shape
    map_t = map_t.to(device=template.device, dtype=template.dtype)
    if map_t.ndim == 2:
        map_t = map_t[None, None]
    elif map_t.ndim == 3:
        if map_t.shape[0] in (1, 3, 4, 6):
            map_t = map_t[None]
        elif map_t.shape[0] != b:
            raise ValueError(f"Unsupported map shape: {tuple(map_t.shape)}")
    elif map_t.ndim != 4:
        raise ValueError(f"Unsupported map shape: {tuple(map_t.shape)}")

    if map_t.shape[-2:] != (h, w):
        raise ValueError(
            f"Map shape {tuple(map_t.shape[-2:])} does not match image shape {(h, w)}"
        )

    if map_t.shape[0] == 1:
        map_t = map_t.expand(b, -1, -1, -1)
    if map_t.shape[1] != channels:
        if map_t.shape[1] == 3 and channels == 1:
            map_t = map_t.mean(dim=1, keepdim=True)
        elif map_t.shape[1] == 1 and channels == 3:
            map_t = map_t.expand(-1, 3, -1, -1)
        elif map_t.shape[1] == 1 and channels == 4:
            map_t = map_t.expand(-1, 4, -1, -1)
        elif map_t.shape[1] == 1 and channels == 6:
            map_t = map_t.expand(-1, 6, -1, -1)
        elif map_t.shape[1] == 6 and channels == 1:
            map_t = map_t.mean(dim=1, keepdim=True)
        elif map_t.shape[1] == 4 and channels == 1:
            map_t = map_t[:, :3].mean(dim=1, keepdim=True)
        elif map_t.shape[1] == 4 and channels == 3:
            map_t = map_t[:, :3]
        elif map_t.shape[1] == 3 and channels == 6:
            map_t = torch.cat([map_t, 1.0 - map_t], dim=1)
        elif map_t.shape[1] == 3 and channels == 4:
            alpha = torch.ones_like(map_t[:, :1])
            map_t = torch.cat([map_t, alpha], dim=1)
        elif map_t.shape[1] == 4 and channels == 6:
            rgb = map_t[:, :3]
            map_t = torch.cat([rgb, 1.0 - rgb], dim=1)
        else:
            raise ValueError(f"Map channels {map_t.shape[1]} incompatible with {channels}")
    return map_t


def dump_tensor_to_txt(
    tensor: Tensor,
    path: Path,
    precision: int = 6,
    binary: bool = False,
    binary_threshold: float = 0.5,
) -> None:
    """Dump a [1, C, H, W] tensor to text (row/col matrix for each channel)."""
    arr = tensor.detach().to(dtype=torch.float32, device="cpu")
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim != 3:
        raise ValueError(f"Expected [1,C,H,W] or [C,H,W] tensor, got {tuple(arr.shape)}")

    if binary:
        arr = (arr > binary_threshold).to(torch.int64)
    else:
        arr = arr.clamp(0.0, 1.0)
    with open(path, "w", encoding="utf-8") as f:
        if binary:
            c_fmt = "{:d}"
        else:
            c_fmt = f"{{:.{precision}f}}"
        for c in range(arr.shape[0]):
            f.write(f"CHANNEL {c}\n")
            ch = arr[c]
            h, w = ch.shape
            for i in range(h):
                row = " ".join(c_fmt.format(int(v.item()) if binary else v.item()) for v in ch[i].flatten())
                f.write(row)
                if i < h - 1:
                    f.write("\n")
            if c < arr.shape[0] - 1:
                f.write("\n\n")


def linear_weights_to_image_tensor(
    image: Tensor,
    linear_mapping: Tensor,
    smooth: int = 15,
    alpha_percentile: float = 99.5,
    eps: float = 1e-12,
) -> Tensor:
    """
    Differentiable tensor version of B-cos gradient_to_image.
    Returns RGBA explanation image tensor [B, 4, H, W] in [0, 1],
    matching bcos.common.gradient_to_image.
    """
    if image.ndim != 4 or linear_mapping.ndim != 4:
        raise ValueError(
            f"Expected image and linear_mapping as [B,C,H,W], got {tuple(image.shape)} and {tuple(linear_mapping.shape)}"
        )
    if image.shape != linear_mapping.shape:
        raise ValueError(
            f"image and linear_mapping shapes must match, got {tuple(image.shape)} and {tuple(linear_mapping.shape)}"
        )
    if image.shape[1] < 6:
        raise ValueError(f"Expected B-cos 6-channel input, got {image.shape[1]} channels")

    contribs = (image * linear_mapping).sum(dim=1, keepdim=True)
    rgb_grad = linear_mapping / (linear_mapping.abs().amax(dim=1, keepdim=True) + eps)
    rgb_grad = rgb_grad.clamp(min=0)
    rgb_grad = rgb_grad[:, :3] / (rgb_grad[:, :3] + rgb_grad[:, 3:6] + eps)

    alpha = linear_mapping.norm(p=2, dim=1, keepdim=True)
    alpha = torch.where(contribs < 0, alpha.new_full((), eps), alpha)
    if smooth:
        alpha = F.avg_pool2d(alpha, smooth, stride=1, padding=(smooth - 1) // 2)
    alpha_scale = torch.quantile(
        alpha.flatten(1),
        q=alpha_percentile / 100,
        dim=1,
        keepdim=True,
    ).view(alpha.shape[0], 1, 1, 1)
    alpha = (alpha / alpha_scale.clamp_min(eps)).clamp(0, 1)

    return torch.cat([rgb_grad, alpha], dim=1)


def _frozen_bcos_conv_forward(module: torch.nn.Module, state: dict, in_tensor: Tensor) -> Tensor:
    """Forward through one BcosConv2d using dynamic scale frozen from a reference input."""
    weight = module.linear.weight
    weight = weight / torch.linalg.vector_norm(weight, dim=(1, 2, 3), keepdim=True)
    out = module.linear._conv_forward(in_tensor, weight, module.linear.bias)

    if module.max_out > 1:
        batch = out.shape[0]
        out = out.unflatten(dim=1, sizes=(module.out_channels, module.max_out))
        max_idx = state["max_idx"].expand(batch, -1, -1, -1)
        out = out.gather(dim=2, index=max_idx.unsqueeze(2)).squeeze(2)

    out = state["scale"] * out
    if hasattr(module, "scale"):
        out = out / module.scale
    return out


def _is_position_norm(module: torch.nn.Module) -> bool:
    name = module.__class__.__name__
    pretty_name = module._get_name() if hasattr(module, "_get_name") else name
    return "PositionNorm" in name or "PositionNorm" in pretty_name


def _frozen_position_norm_forward(module: torch.nn.Module, state: dict, in_tensor: Tensor) -> Tensor:
    if state["centered"]:
        mean = in_tensor.mean(dim=1, keepdim=True)
        out = (in_tensor - mean) / state["std"]
    else:
        out = in_tensor / state["std"]

    if module.weight is not None:
        out = module.weight[None, ..., None, None] * out
    if module.bias is not None:
        out = out + module.bias[None, ..., None, None]
    return out


def _register_linearization_hooks(model: torch.nn.Module) -> Tuple[dict, list]:
    from bcos.modules import BcosConv2d

    states: dict[torch.nn.Module, dict[str, Tensor | None | bool]] = {}
    hooks = []

    def save_bcos_state(module: torch.nn.Module, inputs) -> None:
        in_tensor = inputs[0]
        linear_out = module.linear(in_tensor)
        max_idx = None
        if module.max_out > 1:
            linear_out = linear_out.unflatten(dim=1, sizes=(module.out_channels, module.max_out))
            linear_out, max_idx = linear_out.max(dim=2)

        if module.b == 1:
            dynamic_scale = torch.ones_like(linear_out)
        else:
            norm = module.calc_patch_norms(in_tensor)
            if module.b == 2:
                dynamic_scale = linear_out.abs() / norm
            else:
                dynamic_scale = ((linear_out / norm).abs() + 1e-6).pow(module.b - 1)

        states[module] = {
            "kind": "bcos",
            "scale": dynamic_scale,
            "max_idx": max_idx,
        }

    def save_position_norm_state(module: torch.nn.Module, inputs) -> None:
        in_tensor = inputs[0]
        centered = module.__class__.__name__.startswith("DetachablePositionNorm")
        if centered:
            var = torch.var(in_tensor, dim=1, unbiased=False, keepdim=True)
        else:
            var = torch.var(in_tensor, dim=1, unbiased=False, keepdim=True)

        states[module] = {
            "kind": "position_norm",
            "centered": centered,
            "std": (var + module.eps).sqrt(),
        }

    for module in model.modules():
        if isinstance(module, BcosConv2d):
            hooks.append(module.register_forward_pre_hook(save_bcos_state))
        elif _is_position_norm(module):
            hooks.append(module.register_forward_pre_hook(save_position_norm_state))

    return states, hooks


def _patch_linearized_forwards(states: dict) -> dict:
    original_forwards = {}
    for module, state in states.items():
        original_forwards[module] = module.forward
        if state["kind"] == "bcos":
            module.forward = lambda in_tensor, module=module, state=state: _frozen_bcos_conv_forward(
                module,
                state,
                in_tensor,
            )
        elif state["kind"] == "position_norm":
            module.forward = lambda in_tensor, module=module, state=state: _frozen_position_norm_forward(
                module,
                state,
                in_tensor,
            )
    return original_forwards


def _restore_forwards(original_forwards: dict) -> None:
    for module, forward in original_forwards.items():
        module.forward = forward


def fast_forward_matrix_linear_weights(
    model: torch.nn.Module,
    x_bcos: Tensor,
    target_class: int,
    create_graph: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Compute the target-class W1->L row with the matrix-free frozen network."""
    if x_bcos.ndim != 4 or x_bcos.shape[0] != 1:
        raise ValueError(f"Expected x_bcos shape [1,C,H,W], got {tuple(x_bcos.shape)}")

    states, hooks = _register_linearization_hooks(model)

    try:
        score = model(x_bcos)[:, target_class].sum()
    finally:
        for hook in hooks:
            hook.remove()

    if not states:
        raise RuntimeError("No linearizable modules found for W1->L weights.")

    original_forwards = _patch_linearized_forwards(states)

    x_linear = x_bcos.clone().requires_grad_(True)
    try:
        linear_score = model(x_linear)[:, target_class].sum()
        weights = torch.autograd.grad(
            outputs=linear_score,
            inputs=x_linear,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=False,
        )[0]
    finally:
        _restore_forwards(original_forwards)

    if weights is None:
        raise RuntimeError("Failed to compute matrix-free W1->L weights.")
    return score, weights


def build_loss(
    weights: Tensor,
    x_bcos: Tensor,
    args: argparse.Namespace,
    target_map: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    rendered_image: Optional[Tensor] = None,
) -> Tensor:
    if args.objective == "match":
        if target_map is None:
            raise ValueError("--objective match requires --target-map")
        if args.match_mode == "manual-image":
            if rendered_image is None:
                rendered_image = linear_weights_to_image_tensor(
                    x_bcos,
                    weights,
                    smooth=args.explain_smooth,
                    alpha_percentile=args.explain_alpha_percentile,
                )
            pred_map = rendered_image
        elif args.match_mode == "weights":
            pred_map = weights
        elif args.match_mode == "contrib":
            pred_map = (x_bcos * weights).abs()
        else:
            raise ValueError(f"Unsupported --match-mode: {args.match_mode}")
        return F.mse_loss(pred_map, target_map)

    active = (x_bcos * weights).abs() if args.objective == "contrib" else weights.abs()
    if mask is not None:
        active = active * mask + args.outside_weight * active * (1.0 - mask)
    return -active.mean()


def perturbation_regularizer(delta: Tensor, reg_norm: str) -> Tensor:
    norm = canonicalize_norm(reg_norm)
    if norm == "l2":
        return delta.pow(2).mean()
    if norm == "linf":
        return delta.abs().amax()
    raise ValueError("--reg-norm supports only l2 or linf/l_inf.")


def perturbation_norm_value(delta: Tensor, norm: str) -> float:
    norm = canonicalize_norm(norm)
    flat = delta.detach().reshape(delta.shape[0], -1)
    if norm == "linf":
        value = flat.abs().amax(dim=1)
    elif norm == "l2":
        value = flat.norm(p=2, dim=1)
    elif norm == "l0":
        value = (flat != 0).sum(dim=1).to(dtype=flat.dtype)
    else:
        raise AssertionError(f"Unhandled norm '{norm}'.")
    return float(value.max().item())


def optimize(
    model: torch.nn.Module,
    add_inverse: "object",
    x_rgb: Tensor,
    target_class: int,
    args: argparse.Namespace,
) -> Tensor:
    delta = torch.zeros_like(x_rgb, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=args.lr)
    norm = canonicalize_norm(args.norm)
    reg_norm = canonicalize_norm(args.reg_norm)
    bcos_channels = to_bcos_input(x_rgb, add_inverse=add_inverse).shape[1]
    target_channels = 4 if args.match_mode == "manual-image" else bcos_channels
    print(f"Perturbation budget: {norm} <= {args.epsilon:g}; regularizer={reg_norm}")

    if args.objective == "match":
        if not args.target_map:
            raise ValueError("--objective match requires --target-map")
        print("Using target map for --objective match:", args.target_map)
        print(f"[match] mode={args.match_mode}; source weights=fast_forward_matrix_linear_weights")
        raw = _load_image_as_tensor(
            args.target_map,
            (x_rgb.shape[2], x_rgb.shape[3]),
            rgba=args.match_mode == "manual-image",
        )
        target_map = normalize_map_like(
            x_rgb.new_empty((1, target_channels, x_rgb.shape[2], x_rgb.shape[3])),
            raw,
            channels=target_channels,
        ).detach()
        if args.match_mode == "manual-image":
            non_black_rgb = target_map[:, :3].abs().sum(dim=1, keepdim=True) > 0
            target_map = torch.cat(
                [
                    target_map[:, :3],
                    target_map[:, 3:4] * non_black_rgb.to(dtype=target_map.dtype),
                ],
                dim=1,
            )
            print("[target] RGB-black pixels use alpha=0.")
        print("Target map shape:", tuple(target_map.shape))
        if args.target_map_txt is not None:
            dump_tensor_to_txt(
                target_map,
                Path(args.target_map_txt),
                binary=args.target_map_txt_binary,
                binary_threshold=args.target_map_txt_threshold,
            )
    else:
        target_map = None

    mask = None
    if args.mask:
        raw = _load_image_as_tensor(args.mask, (x_rgb.shape[2], x_rgb.shape[3]))
        mask = normalize_map_like(x_rgb, raw, channels=1).clamp(0.0, 1.0).detach()

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        x_adv = (x_rgb + delta).clamp(0.0, 1.0)
        x_adv_bcos = to_bcos_input(x_adv, add_inverse=add_inverse)
        _, weights = fast_forward_matrix_linear_weights(
            model=model,
            x_bcos=x_adv_bcos,
            target_class=target_class,
            create_graph=True,
        )

        rendered_image = None
        if args.objective == "match" and args.match_mode == "manual-image":
            rendered_image = linear_weights_to_image_tensor(
                x_adv_bcos,
                weights,
                smooth=args.explain_smooth,
                alpha_percentile=args.explain_alpha_percentile,
            )

        loss_map = build_loss(
            weights=weights,
            x_bcos=x_adv_bcos,
            args=args,
            target_map=target_map,
            mask=mask,
            rendered_image=rendered_image,
        )
        loss = loss_map + args.reg * perturbation_regularizer(delta, reg_norm)
        loss.backward()
        loss_item = float(loss.detach().item())

        if rendered_image is not None:
            pred_mag = float(rendered_image.detach().abs().mean().item())
        elif args.objective == "contrib" or (
            args.objective == "match" and args.match_mode == "contrib"
        ):
            pred_mag = float((x_adv_bcos * weights).detach().abs().mean().item())
        else:
            pred_mag = float(weights.detach().abs().mean().item())

        optimizer.step()

        with torch.no_grad():
            delta.copy_(project_perturbation(delta, args.epsilon, norm))
            delta_norm = perturbation_norm_value(delta, norm)

        if step % 10 == 0 or step == 1:
            with torch.no_grad():
                score = model(to_bcos_input(x_adv, add_inverse))[0, target_class].item()
            print(
                f"Step {step:4d}/{args.steps} | "
                f"loss={loss_item:.4f} | pred|avg|={pred_mag:.4f} | "
                f"delta_{norm}={delta_norm:.4f} | score(target)={score:.4f}"
            )

    return (x_rgb + delta).clamp(0.0, 1.0).detach()


def save_rgb_image(image_tensor: Tensor, path: Path) -> None:
    """Save [1,3,H,W] tensor in [0,1] to PNG/JPG."""
    x = image_tensor.detach().clamp(0.0, 1.0).cpu()[0].permute(1, 2, 0).numpy()
    x = (x * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(x).save(str(path))


def extract_explanation(
    model: torch.nn.Module,
    add_inverse: "object",
    x_rgb: Tensor,
    target_class: int,
    smooth: int = 15,
    alpha_percentile: float = 99.5,
) -> np.ndarray:
    """Render an RGBA explanation from fast matrix-free W1->L weights."""
    x_6ch = to_bcos_input(x_rgb, add_inverse).detach().clone()
    _, weights = fast_forward_matrix_linear_weights(
        model=model,
        x_bcos=x_6ch,
        target_class=target_class,
        create_graph=False,
    )
    rgba = linear_weights_to_image_tensor(
        x_6ch,
        weights,
        smooth=smooth,
        alpha_percentile=alpha_percentile,
    )
    return rgba[0].detach().cpu().permute(1, 2, 0).numpy()


def save_explain_map(explanation: np.ndarray, path: Path) -> None:
    """Save rendered RGBA explanation output."""
    arr = np.asarray(explanation)
    arr = np.nan_to_num(arr)
    if arr.ndim != 3 or arr.shape[-1] != 4:
        raise ValueError(f"Unexpected explain map shape: {arr.shape}")
    arr = arr.clip(0.0, 1.0)
    if arr.dtype != np.uint8:
        arr = (arr * 255.0).round().astype(np.uint8)
    Image.fromarray(arr, mode="RGBA").save(str(path))


def build_target_map(
    model: torch.nn.Module,
    add_inverse: "object",
    x_rgb: Tensor,
    target_class: int,
    source: str = "manual-image",
    smooth: int = 15,
    alpha_percentile: float = 99.5,
) -> Tensor:
    """
    Create a target map from fast matrix-free W1->L weights.

    Sources:
      - manual-image: rendered RGBA explanation image [1,4,H,W]
      - weights: raw W1->L weights [1,6,H,W]
      - contribution: signed sum_c(x_c * W_c) [1,1,H,W]
    """
    x_6ch = to_bcos_input(x_rgb, add_inverse=add_inverse).detach().clone()
    _, weights = fast_forward_matrix_linear_weights(
        model=model,
        x_bcos=x_6ch,
        target_class=target_class,
        create_graph=False,
    )

    if source == "manual-image":
        map_t = linear_weights_to_image_tensor(
            x_6ch,
            weights,
            smooth=smooth,
            alpha_percentile=alpha_percentile,
        )
    elif source == "weights":
        map_t = weights
    elif source == "contribution":
        map_t = (x_6ch * weights).sum(dim=1, keepdim=True)
    else:
        raise ValueError(f"Unsupported target map source: {source}")

    if map_t.ndim == 2:
        map_t = map_t[None, None]
    elif map_t.ndim == 3:
        map_t = map_t[None]
    elif map_t.ndim != 4:
        raise ValueError(f"Unexpected target map shape: {tuple(map_t.shape)}")

    if map_t.shape[0] != 1:
        map_t = map_t[:1]

    if source == "manual-image":
        return map_t.detach().to(dtype=x_rgb.dtype, device=x_rgb.device).clamp(0.0, 1.0)

    map_t = map_t.detach().to(dtype=x_rgb.dtype, device=x_rgb.device)
    map_t = map_t - map_t.amin(dim=(-1, -2), keepdim=True)
    max_hw = map_t.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
    return map_t / max_hw


def save_target_image(map_t: Tensor, path: Path) -> None:
    """Save target map [1,C,H,W] to PNG, or to NPY when preserving channels matters."""
    if map_t.ndim != 4 or map_t.shape[0] != 1:
        raise ValueError(f"Unexpected target map shape for saving: {tuple(map_t.shape)}")
    if path.suffix.lower() == ".npy":
        arr = map_t[0].detach().to(dtype=torch.float32, device="cpu").numpy()
        np.save(str(path), arr)
        return
    map_vis = map_t[0]
    if map_vis.shape[0] in (3, 4):
        arr = map_vis.detach().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
        arr = (arr * 255.0).round().clip(0, 255).astype(np.uint8)
        Image.fromarray(arr, mode="RGBA" if map_vis.shape[0] == 4 else "RGB").save(str(path))
        return
    if map_vis.shape[0] != 1:
        map_vis = map_vis.abs().mean(0, keepdim=True)
    arr = map_vis[0].detach().clamp(0.0, 1.0).cpu().numpy()
    arr = (arr * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(str(path))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="B-cos white-box attack using fast matrix-free W1->L weights."
    )
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--model", default="resnet18", help="B-cos pretrained model entrypoint")
    p.add_argument("--target-class", type=int, default=None, help="Target class index")
    p.add_argument("--steps", type=int, default=120, help="Optimization steps")
    p.add_argument("--lr", type=float, default=5e-2, help="Optimizer learning rate")
    p.add_argument("--epsilon", type=float, default=0.15, help="Perturbation budget")
    p.add_argument(
        "--norm",
        default="linf",
        help="Perturbation budget norm: l0, l2, or linf. l_inf is accepted as an alias.",
    )
    p.add_argument("--reg", type=float, default=1, help="Regularization weight on delta")
    p.add_argument(
        "--reg-norm",
        default="l2",
        help="Regularizer norm on delta: l2 or linf. l_inf is accepted as an alias.",
    )
    p.add_argument("--outside-weight", type=float, default=0.1, help="Penalty outside mask")
    p.add_argument(
        "--objective",
        default="abs",
        choices=("abs", "contrib", "match"),
        help="abs=max |W1->L|, contrib=max |x*W1->L|, match=MSE with target using --match-mode",
    )
    p.add_argument(
        "--match-mode",
        default="manual-image",
        choices=("manual-image", "weights", "contrib"),
        help=(
            "For --objective match: compare rendered RGBA W1->L image (manual-image), "
            "raw W1->L weights (weights), or |x*W1->L| (contrib)."
        ),
    )
    p.add_argument(
        "--explain-smooth",
        type=int,
        default=15,
        help="Smoothing kernel used when rendering Manual W1->L Image.",
    )
    p.add_argument(
        "--explain-alpha-percentile",
        type=float,
        default=99.5,
        help="Alpha percentile used when rendering Manual W1->L Image.",
    )
    p.add_argument("--target-map", default=None, help="Target map for --objective match")
    p.add_argument("--target-map-txt", default=None, help="Path to dump target map as text")
    p.add_argument(
        "--target-map-txt-binary",
        action="store_true",
        help="Dump target map text as 0/1 only",
    )
    p.add_argument(
        "--target-map-txt-threshold",
        type=float,
        default=0.5,
        help="Threshold for --target-map-txt-binary (default: 0.5)",
    )
    p.add_argument("--mask", default=None, help="Optional mask for abs/contrib objective")
    p.add_argument("--create-target", action="store_true", help="Create target map image and exit")
    p.add_argument(
        "--target-source",
        default="manual-image",
        choices=("manual-image", "weights", "contribution"),
        help="Source used by --create-target. All sources use fast matrix-free W1->L weights.",
    )
    p.add_argument("--target-image", default="target_map.png", help="Output target map path")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--device", default="auto", help="auto|cpu|cuda")
    p.add_argument("--save", default="whitebox_explain_result.png", help="Save path")
    p.add_argument("--explain-before", default=None, help="Path to save explain map before attack")
    p.add_argument("--explain-after", default=None, help="Path to save explain map after attack")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_sys_path()
    import bcos as bcos_pkg
    from bcos import transforms as bcos_transforms

    model_fn = getattr(bcos_pkg.pretrained, args.model)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = model_fn(pretrained=True).to(device).eval()

    for p in model.parameters():
        p.requires_grad_(False)

    add_inverse = bcos_transforms.AddInverse()

    x_rgb = load_rgb_input(args.image, model, device=device)
    with torch.no_grad():
        out = model(to_bcos_input(x_rgb, add_inverse))
        pred = int(out.argmax(1).item())
        target_class = pred if args.target_class is None else args.target_class
        target_logit = float(out[0, target_class].item())

    if args.target_class is None:
        print(f"[auto] Target class = {target_class}, original logit={target_logit:.4f}")
    else:
        print(
            f"Using class {target_class} (original logit={target_logit:.4f} for original prediction {pred})"
        )

    if args.create_target:
        target_map = build_target_map(
            model=model,
            add_inverse=add_inverse,
            x_rgb=x_rgb,
            target_class=target_class,
            source=args.target_source,
            smooth=args.explain_smooth,
            alpha_percentile=args.explain_alpha_percentile,
        )
        target_path = Path(args.target_image)
        os.makedirs(target_path.parent, exist_ok=True)
        save_target_image(target_map, target_path)
        print(f"Saved target map: {target_path}")
        if args.target_map_txt is not None:
            dump_path = Path(args.target_map_txt)
            os.makedirs(dump_path.parent, exist_ok=True)
            dump_tensor_to_txt(
                target_map,
                dump_path,
                binary=args.target_map_txt_binary,
                binary_threshold=args.target_map_txt_threshold,
            )
            print(f"Saved target map text: {dump_path}")
        return

    explain_before = extract_explanation(
        model=model,
        add_inverse=add_inverse,
        x_rgb=x_rgb,
        target_class=target_class,
        smooth=args.explain_smooth,
        alpha_percentile=args.explain_alpha_percentile,
    )

    x_adv = optimize(
        model=model,
        add_inverse=add_inverse,
        x_rgb=x_rgb,
        target_class=target_class,
        args=args,
    )

    out_path = Path(args.save)
    os.makedirs(out_path.parent, exist_ok=True)
    out_dir = out_path.parent
    stem = out_path.stem if out_path.stem else "whitebox_explain"

    explain_before_path = (
        Path(args.explain_before)
        if args.explain_before is not None
        else out_dir / f"{stem}_explain_before.png"
    )
    explain_after_path = (
        Path(args.explain_after)
        if args.explain_after is not None
        else out_dir / f"{stem}_explain_after.png"
    )

    save_explain_map(explain_before, explain_before_path)
    explain_after = extract_explanation(
        model=model,
        add_inverse=add_inverse,
        x_rgb=x_adv,
        target_class=target_class,
        smooth=args.explain_smooth,
        alpha_percentile=args.explain_alpha_percentile,
    )
    save_explain_map(explain_after, explain_after_path)

    save_rgb_image(x_adv, out_path)
    print(f"Saved image: {out_path}")
    print(f"Saved explain before: {explain_before_path}")
    print(f"Saved explain after: {explain_after_path}")


if __name__ == "__main__":
    main()
