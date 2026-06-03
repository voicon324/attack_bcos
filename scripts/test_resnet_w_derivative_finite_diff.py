#!/usr/bin/env python3
"""
Numerically check dW(x)/dx on a real B-cos ResNet.

For the full ResNet, writing out the full product-rule formula by hand is too
large to be useful. Instead, this script checks the same mathematical claim in
a practical way:

    phi(x) = one selected coordinate of W_c(x)

Then it compares:

    d phi / d x_j from autograd

against:

    [phi(x + h e_j) - phi(x - h e_j)] / (2h)

where W_c(x) is computed by the repo's fast_forward_matrix_linear_weights().
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTACKS_DIR = PROJECT_ROOT / "attacks"
BCOS_DIR = PROJECT_ROOT / "B-cos-v2"

sys.path.insert(0, str(ATTACKS_DIR))
sys.path.insert(0, str(BCOS_DIR))

from whitebox_w1l_attack import fast_forward_matrix_linear_weights  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finite-difference check of dW/dx for a B-cos ResNet.",
    )
    parser.add_argument("--model", default="resnet18", help="B-cos ResNet name.")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load bcos.pretrained.<model>(pretrained=True). Default uses random ResNet weights.",
    )
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N.")
    parser.add_argument("--height", type=int, default=16, help="Input height.")
    parser.add_argument("--width", type=int, default=16, help="Input width.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--checks", type=int, default=4, help="Number of input coordinates to test.")
    parser.add_argument("--fd-step", type=float, default=1e-5, help="Finite-difference step h.")
    parser.add_argument("--target-class", type=int, default=None, help="Class index. Default: model prediction.")
    parser.add_argument(
        "--w-index",
        type=int,
        default=None,
        help="Flattened W coordinate to test. Default: random coordinate.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="torch.set_num_threads value for CPU runs.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
        help="Computation dtype. float64 gives a cleaner finite-difference check on CPU.",
    )
    parser.add_argument("--rtol", type=float, default=5e-2, help="Relative tolerance.")
    parser.add_argument("--atol", type=float, default=5e-4, help="Absolute tolerance.")
    return parser.parse_args()


def load_resnet_model(
    name: str,
    pretrained: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    if pretrained:
        import bcos

        model_fn = getattr(bcos.pretrained, name)
        model = model_fn(pretrained=True)
    else:
        from bcos.models import resnet as resnet_models

        model_fn = getattr(resnet_models, name)
        model = model_fn(pretrained=False)

    model = model.to(device=device, dtype=dtype).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def make_valid_bcos_input(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Create a valid [rgb, 1-rgb] B-cos input with shape [1, 6, H, W]."""
    rgb = torch.rand(1, 3, height, width, device=device, dtype=dtype) * 0.8 + 0.1
    return torch.cat([rgb, 1.0 - rgb], dim=1)


def compute_w(
    model: torch.nn.Module,
    x_bcos: Tensor,
    target_class: int,
    create_graph: bool,
) -> Tensor:
    _, weights = fast_forward_matrix_linear_weights(
        model=model,
        x_bcos=x_bcos,
        target_class=target_class,
        create_graph=create_graph,
    )
    return weights


def phi_at(
    model: torch.nn.Module,
    x_bcos: Tensor,
    target_class: int,
    w_index: int,
) -> Tensor:
    w = compute_w(
        model=model,
        x_bcos=x_bcos,
        target_class=target_class,
        create_graph=False,
    )
    return w.reshape(-1)[w_index]


def choose_indices(total: int, count: int, protected: Iterable[int]) -> list[int]:
    protected_set = set(protected)
    candidates = [idx for idx in range(total) if idx not in protected_set]
    return random.sample(candidates, k=min(count, len(candidates)))


def main() -> None:
    args = parse_args()

    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be positive.")
    if args.checks <= 0:
        raise ValueError("--checks must be positive.")
    if args.fd_step <= 0:
        raise ValueError("--fd-step must be positive.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_printoptions(precision=8, sci_mode=False)

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    print("=== Loading model ===")
    print(f"model      : {args.model}")
    print(f"pretrained : {args.pretrained}")
    print(f"device     : {device}")
    print(f"input      : [1, 6, {args.height}, {args.width}]")
    model = load_resnet_model(args.model, args.pretrained, device, dtype)

    x = make_valid_bcos_input(args.height, args.width, device=device, dtype=dtype)

    with torch.no_grad():
        logits = model(x)
        pred_class = int(logits.argmax(dim=1).item())
    target_class = pred_class if args.target_class is None else int(args.target_class)

    print(f"pred class : {pred_class}")
    print(f"test class : {target_class}")
    print()

    x_for_grad = x.detach().clone().requires_grad_(True)
    w = compute_w(
        model=model,
        x_bcos=x_for_grad,
        target_class=target_class,
        create_graph=True,
    )
    w_flat = w.reshape(-1)
    x_flat = x_for_grad.reshape(-1)

    if args.w_index is None:
        w_index = random.randrange(w_flat.numel())
    else:
        w_index = int(args.w_index)
    if not 0 <= w_index < w_flat.numel():
        raise ValueError(f"--w-index must be in [0, {w_flat.numel() - 1}].")

    phi = w_flat[w_index]
    grad_phi = torch.autograd.grad(
        outputs=phi,
        inputs=x_for_grad,
        retain_graph=False,
        create_graph=False,
    )[0].reshape(-1)

    x_indices = choose_indices(
        total=x_flat.numel(),
        count=args.checks,
        protected=[w_index] if w_index < x_flat.numel() else [],
    )

    print("=== Testing scalar projection ===")
    print(f"phi(x) = W_flat[{w_index}]")
    print(f"phi value = {float(phi.detach().item()):.8g}")
    print()

    rows = []
    h = args.fd_step
    base_x = x.detach()

    for x_index in x_indices:
        direction = torch.zeros_like(base_x).reshape(-1)
        direction[x_index] = h
        direction = direction.reshape_as(base_x)

        phi_plus = phi_at(
            model=model,
            x_bcos=base_x + direction,
            target_class=target_class,
            w_index=w_index,
        )
        phi_minus = phi_at(
            model=model,
            x_bcos=base_x - direction,
            target_class=target_class,
            w_index=w_index,
        )

        finite_diff = (phi_plus - phi_minus) / (2.0 * h)
        autograd_value = grad_phi[x_index]
        abs_error = (autograd_value - finite_diff).abs()
        denom = finite_diff.abs().clamp_min(1e-12)
        rel_error = abs_error / denom
        ok = bool(abs_error <= args.atol + args.rtol * finite_diff.abs())

        rows.append((x_index, autograd_value, finite_diff, abs_error, rel_error, ok))

    print("=== d phi / d x_j comparison ===")
    all_ok = True
    for x_index, autograd_value, finite_diff, abs_error, rel_error, ok in rows:
        all_ok = all_ok and ok
        print(
            f"x_flat[{x_index:5d}]  "
            f"autograd={float(autograd_value): .8e}  "
            f"finite_diff={float(finite_diff): .8e}  "
            f"abs_err={float(abs_error):.3e}  "
            f"rel_err={float(rel_error):.3e}  "
            f"ok={ok}"
        )

    print()
    if not all_ok:
        raise SystemExit("FAIL: at least one finite-difference check did not match.")
    print("PASS: ResNet dW/dx autograd matches finite differences for the tested coordinates.")


if __name__ == "__main__":
    main()
