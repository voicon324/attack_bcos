#!/usr/bin/env python3
"""
Activation maximization for a node/channel in a B-cos ResNet.

The script optimizes an RGB image x so that a selected layer/channel activation
is large. It answers the question:

    "This node wants to see what kind of input?"

The optimized RGB image is converted to B-cos input as:

    [rgb, 1 - rgb]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BCOS_DIR = PROJECT_ROOT / "B-cos-v2"
DEFAULT_OUT_DIR = PROJECT_ROOT / "artifacts" / "outputs" / "node_activation_max"

sys.path.insert(0, str(BCOS_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize an input image that maximally activates one B-cos ResNet node.",
    )
    parser.add_argument("--model", default="resnet18", help="B-cos model name.")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load pretrained B-cos weights. Default uses random model weights.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--height", type=int, default=128, help="Optimized image height.")
    parser.add_argument("--width", type=int, default=128, help="Optimized image width.")
    parser.add_argument(
        "--layer",
        default="layer4.1.conv2",
        help="Module name to optimize. Use --list-layers to inspect names.",
    )
    parser.add_argument("--channel", type=int, default=0, help="Channel/node index inside the layer output.")
    parser.add_argument(
        "--objective",
        choices=("node", "class-logit", "class-margin"),
        default="node",
        help=(
            "node optimizes the selected layer/channel activation. "
            "class-logit optimizes final model logit/class channel and ignores --layer. "
            "class-margin optimizes target logit against all other classes."
        ),
    )
    parser.add_argument(
        "--spatial",
        choices=("mean", "max", "center", "yx"),
        default="max",
        help="How to reduce spatial positions for the selected channel.",
    )
    parser.add_argument("--pos-y", type=int, default=0, help="Y position when --spatial yx.")
    parser.add_argument("--pos-x", type=int, default=0, help="X position when --spatial yx.")
    parser.add_argument("--steps", type=int, default=300, help="Optimization steps.")
    parser.add_argument("--lr", type=float, default=0.05, help="Adam learning rate.")
    parser.add_argument("--tv-weight", type=float, default=2e-4, help="Total-variation regularization weight.")
    parser.add_argument("--l2-weight", type=float, default=1e-4, help="Small RGB L2 regularization weight.")
    parser.add_argument(
        "--octaves",
        type=int,
        default=1,
        help="Optimize from low resolution to final resolution. Higher values usually look cleaner.",
    )
    parser.add_argument(
        "--blur-every",
        type=int,
        default=0,
        help="Apply Gaussian blur to the image every N steps. Use 0 to disable.",
    )
    parser.add_argument("--blur-sigma", type=float, default=0.8, help="Gaussian blur sigma.")
    parser.add_argument(
        "--jitter",
        type=int,
        default=8,
        help="Random translation jitter in pixels. Use 0 to disable.",
    )
    parser.add_argument(
        "--init",
        choices=("random", "gray"),
        default="random",
        help="Initial image.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--print-every", type=int, default=25, help="Progress print interval.")
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="Print candidate layer names and exit.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_model(name: str, pretrained: bool, device: torch.device) -> torch.nn.Module:
    if pretrained:
        import bcos

        model_fn = getattr(bcos.pretrained, name)
        model = model_fn(pretrained=True)
    else:
        from bcos.models import resnet as resnet_models

        model_fn = getattr(resnet_models, name)
        model = model_fn(pretrained=False)

    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def candidate_layer_names(model: torch.nn.Module) -> list[str]:
    names = []
    for name, module in model.named_modules():
        if not name:
            continue
        class_name = module.__class__.__name__
        if class_name in {"BcosConv2d", "DetachablePositionNorm2d", "PositionNormUncentered2d"}:
            names.append(name)
    return names


def get_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    modules = dict(model.named_modules())
    if name not in modules:
        available = "\n".join(candidate_layer_names(model)[:80])
        raise ValueError(
            f"Layer {name!r} not found. Run --list-layers. First candidates:\n{available}"
        )
    return modules[name]


def logit(p: Tensor, eps: float = 1e-5) -> Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


def make_initial_param(args: argparse.Namespace, device: torch.device, height: int, width: int) -> Tensor:
    if args.init == "gray":
        rgb = torch.full((1, 3, height, width), 0.5, device=device)
        rgb = rgb + 0.05 * torch.randn_like(rgb)
        rgb = rgb.clamp(0.0, 1.0)
    else:
        rgb = torch.rand(1, 3, height, width, device=device) * 0.7 + 0.15
    return logit(rgb).detach().requires_grad_(True)


def to_bcos_input(rgb: Tensor) -> Tensor:
    return torch.cat([rgb, 1.0 - rgb], dim=1)


def total_variation(rgb: Tensor) -> Tensor:
    dy = (rgb[:, :, 1:, :] - rgb[:, :, :-1, :]).abs().mean()
    dx = (rgb[:, :, :, 1:] - rgb[:, :, :, :-1]).abs().mean()
    return dx + dy


def gaussian_blur(rgb: Tensor, sigma: float) -> Tensor:
    if sigma <= 0:
        return rgb
    radius = max(1, int(round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=rgb.device, dtype=rgb.dtype)
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    channels = rgb.shape[1]
    kernel_x = kernel_1d.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kernel_y = kernel_1d.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    out = F.conv2d(rgb, kernel_x, padding=(0, radius), groups=channels)
    out = F.conv2d(out, kernel_y, padding=(radius, 0), groups=channels)
    return out


def octave_sizes(height: int, width: int, octaves: int) -> list[tuple[int, int]]:
    sizes = []
    for idx in range(octaves - 1, -1, -1):
        scale = 2**idx
        sizes.append((max(16, height // scale), max(16, width // scale)))
    sizes[-1] = (height, width)

    unique_sizes = []
    for size in sizes:
        if not unique_sizes or unique_sizes[-1] != size:
            unique_sizes.append(size)
    return unique_sizes


def steps_per_octave(total_steps: int, count: int) -> list[int]:
    base = total_steps // count
    extra = total_steps % count
    return [base + (1 if idx < extra else 0) for idx in range(count)]


def reduce_activation(
    activation: Tensor,
    channel: int,
    spatial: str,
    pos_y: int,
    pos_x: int,
) -> Tensor:
    if activation.ndim == 2:
        if not 0 <= channel < activation.shape[1]:
            raise ValueError(f"Channel {channel} out of range for shape {tuple(activation.shape)}.")
        return activation[:, channel].mean()

    if activation.ndim != 4:
        raise ValueError(f"Expected layer activation shape [B,C,H,W] or [B,C], got {tuple(activation.shape)}.")

    _, channels, height, width = activation.shape
    if not 0 <= channel < channels:
        raise ValueError(f"Channel {channel} out of range for activation with {channels} channels.")

    feature = activation[:, channel]
    if spatial == "mean":
        return feature.mean()
    if spatial == "max":
        return feature.flatten(1).max(dim=1).values.mean()
    if spatial == "center":
        return feature[:, height // 2, width // 2].mean()
    if spatial == "yx":
        if not (0 <= pos_y < height and 0 <= pos_x < width):
            raise ValueError(f"Position ({pos_y}, {pos_x}) out of range for activation map {height}x{width}.")
        return feature[:, pos_y, pos_x].mean()
    raise AssertionError(f"Unhandled spatial mode {spatial!r}.")


def save_rgb_image(rgb: Tensor, path: Path) -> None:
    arr = rgb.detach().clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_heatmap(feature: Tensor, path: Path) -> None:
    fmap = feature.detach().float().cpu()
    fmap = fmap - fmap.min()
    denom = fmap.max().clamp_min(1e-12)
    fmap = fmap / denom
    arr = (fmap.numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).resize((256, 256), Image.Resampling.NEAREST).save(path)


def safe_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def main() -> None:
    args = parse_args()
    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be positive.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.jitter < 0:
        raise ValueError("--jitter must be >= 0.")
    if args.octaves <= 0:
        raise ValueError("--octaves must be positive.")
    if args.blur_every < 0:
        raise ValueError("--blur-every must be >= 0.")
    if args.blur_sigma < 0:
        raise ValueError("--blur-sigma must be >= 0.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    model = load_model(args.model, args.pretrained, device)

    if args.list_layers:
        for name in candidate_layer_names(model):
            module = get_module(model, name)
            print(f"{name:35s} {module.__class__.__name__}")
        return

    target_module = get_module(model, args.layer) if args.objective == "node" else None
    captured: dict[str, Tensor] = {}

    def save_activation(_module, _inputs, output) -> None:
        captured["activation"] = output

    hook = target_module.register_forward_hook(save_activation) if target_module is not None else None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    layer_tag = safe_tag(args.layer) if args.objective == "node" else "final_logit"
    tag = (
        f"{safe_tag(args.model)}_{args.objective}_{layer_tag}_ch{args.channel}_"
        f"{args.spatial}_{args.height}x{args.width}"
    )
    out_image = args.out_dir / f"{tag}.png"
    out_heatmap = args.out_dir / f"{tag}_activation.png"

    best_score = -float("inf")
    best_rgb: Optional[Tensor] = None
    start_time = time.perf_counter()

    print("=== Activation maximization ===")
    print(f"model      : {args.model}")
    print(f"pretrained : {args.pretrained}")
    if target_module is not None:
        print(f"layer      : {args.layer} ({target_module.__class__.__name__})")
    else:
        print("layer      : final model logit")
    print(f"channel    : {args.channel}")
    print(f"objective  : {args.objective}")
    print(f"spatial    : {args.spatial}")
    print(f"image      : {args.height}x{args.width}")
    print(f"octaves    : {args.octaves}")
    if args.blur_every:
        print(f"blur       : every {args.blur_every} steps, sigma={args.blur_sigma:g}")
    print(f"device     : {device}")
    print()

    try:
        sizes = octave_sizes(args.height, args.width, args.octaves)
        octave_steps = steps_per_octave(args.steps, len(sizes))
        param: Optional[Tensor] = None
        global_step = 0

        for octave_idx, ((height, width), step_count) in enumerate(zip(sizes, octave_steps), start=1):
            if param is None:
                param = make_initial_param(args, device, height, width)
            else:
                with torch.no_grad():
                    rgb = torch.sigmoid(param)
                    rgb = F.interpolate(rgb, size=(height, width), mode="bilinear", align_corners=False)
                param = logit(rgb).detach().requires_grad_(True)

            optimizer = torch.optim.Adam([param], lr=args.lr)
            print(f"-- octave {octave_idx}/{len(sizes)}: {height}x{width}, steps={step_count}")

            for _ in range(step_count):
                global_step += 1
                optimizer.zero_grad(set_to_none=True)
                rgb = torch.sigmoid(param)

                if args.jitter:
                    shift_y = random.randint(-args.jitter, args.jitter)
                    shift_x = random.randint(-args.jitter, args.jitter)
                    rgb_for_model = torch.roll(rgb, shifts=(shift_y, shift_x), dims=(-2, -1))
                else:
                    rgb_for_model = rgb

                captured.clear()
                logits = model(to_bcos_input(rgb_for_model))
                if args.objective in {"class-logit", "class-margin"}:
                    if not 0 <= args.channel < logits.shape[1]:
                        raise ValueError(f"Class channel {args.channel} out of range for logits {tuple(logits.shape)}.")
                    target_logit = logits[:, args.channel]
                    if args.objective == "class-logit":
                        activation = target_logit.mean()
                    else:
                        other_logits = logits.clone()
                        other_logits[:, args.channel] = -float("inf")
                        activation = (target_logit - torch.logsumexp(other_logits, dim=1)).mean()
                else:
                    if "activation" not in captured:
                        raise RuntimeError(f"Forward hook for layer {args.layer!r} did not capture an activation.")
                    activation = reduce_activation(
                        captured["activation"],
                        channel=args.channel,
                        spatial=args.spatial,
                        pos_y=args.pos_y,
                        pos_x=args.pos_x,
                    )

                tv = total_variation(rgb)
                l2 = (rgb - 0.5).pow(2).mean()
                loss = -activation + args.tv_weight * tv + args.l2_weight * l2
                loss.backward()
                optimizer.step()

                if args.blur_every and global_step % args.blur_every == 0:
                    with torch.no_grad():
                        blurred = gaussian_blur(torch.sigmoid(param), args.blur_sigma).clamp(0.0, 1.0)
                        param.copy_(logit(blurred))

                score = float(activation.detach().item())
                if score > best_score:
                    best_score = score
                    best_rgb = rgb.detach().clone()
                    if best_rgb.shape[-2:] != (args.height, args.width):
                        best_rgb = F.interpolate(
                            best_rgb,
                            size=(args.height, args.width),
                            mode="bilinear",
                            align_corners=False,
                        )

                if global_step == 1 or global_step == args.steps or global_step % args.print_every == 0:
                    print(
                        f"step {global_step:04d}/{args.steps}  "
                        f"activation={score:.6g}  tv={float(tv.detach().item()):.6g}  "
                        f"l2={float(l2.detach().item()):.6g}"
                    )
    finally:
        if hook is not None:
            hook.remove()

    if best_rgb is None:
        raise RuntimeError("Optimization did not produce an image.")

    save_rgb_image(best_rgb, out_image)

    # Save final selected-channel activation map for the best image.
    if target_module is not None:
        captured.clear()
        hook = target_module.register_forward_hook(save_activation)
        try:
            with torch.no_grad():
                _ = model(to_bcos_input(best_rgb))
                final_activation = captured["activation"]
            if final_activation.ndim == 4:
                save_heatmap(final_activation[0, args.channel], out_heatmap)
        finally:
            hook.remove()

    elapsed = time.perf_counter() - start_time
    print()
    print(f"best activation : {best_score:.8g}")
    print(f"saved image     : {out_image}")
    if out_heatmap.exists():
        print(f"saved heatmap   : {out_heatmap}")
    print(f"elapsed         : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
