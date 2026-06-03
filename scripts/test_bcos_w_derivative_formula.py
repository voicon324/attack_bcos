#!/usr/bin/env python3
"""
Check the hand-derived B-cos W(x) and dW/dx formulas against the repo code.

This script builds the smallest possible B-cos model:

    output = s(x) * a^T x

with:

    s(x) = |a^T x| / sqrt(x^T x + eps)
    W(x) = s(x) * a

Then it compares:

1. W(x) from the hand formula
2. W(x) from fast_forward_matrix_linear_weights()
3. dW/dx from the hand formula
4. dW/dx from PyTorch autograd through fast_forward_matrix_linear_weights()
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import Tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTACKS_DIR = PROJECT_ROOT / "attacks"
BCOS_DIR = PROJECT_ROOT / "B-cos-v2"

sys.path.insert(0, str(ATTACKS_DIR))
sys.path.insert(0, str(BCOS_DIR))

from whitebox_w1l_attack import fast_forward_matrix_linear_weights  # noqa: E402
from bcos.modules import BcosConv2d  # noqa: E402


def normalized_weight(raw_weight: Tensor) -> Tensor:
    """Return normalized linear weight a."""
    return raw_weight / torch.linalg.vector_norm(raw_weight)


def manual_bcos_scale(a: Tensor, x: Tensor, eps: float) -> Tensor:
    """s(x) = |a^T x| / sqrt(x^T x + eps)."""
    r = torch.dot(a, x)
    n = torch.sqrt(torch.dot(x, x) + eps)
    return r.abs() / n


def manual_w(a: Tensor, x: Tensor, eps: float) -> Tensor:
    """W(x) = s(x) * a."""
    return manual_bcos_scale(a, x, eps) * a


def manual_ds_dx(a: Tensor, x: Tensor, eps: float) -> Tensor:
    """
    grad_x s(x), where:

        s(x) = |a^T x| / sqrt(x^T x + eps)
    """
    r = torch.dot(a, x)
    n = torch.sqrt(torch.dot(x, x) + eps)
    return torch.sign(r) * a / n - r.abs() * x / n.pow(3)


def manual_d_w_dx(a: Tensor, x: Tensor, eps: float) -> Tensor:
    """
    Jacobian of W(x) = s(x) * a.

    Entry [i, j] is d W_i / d x_j.
    """
    return torch.outer(a, manual_ds_dx(a, x, eps))


class TinyBcosModel(torch.nn.Module):
    """One BcosConv2d unit used as a one-class model."""

    def __init__(self, raw_weight: Tensor) -> None:
        super().__init__()
        out_channels, in_channels, height, width = raw_weight.shape
        if (out_channels, height, width) != (1, 1, 1):
            raise ValueError("raw_weight must have shape [1, C, 1, 1].")

        self.bcos = BcosConv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=1,
            b=2,
            max_out=1,
            dtype=raw_weight.dtype,
        )
        with torch.no_grad():
            self.bcos.linear.weight.copy_(raw_weight)

    def forward(self, x: Tensor) -> Tensor:
        # Input shape: [B, C, 1, 1]
        # Output shape expected by fast_forward_matrix_linear_weights: [B, classes]
        return self.bcos(x).flatten(1)


def code_w_and_d_w_dx(model: torch.nn.Module, x: Tensor) -> tuple[Tensor, Tensor]:
    """Get W(x) and dW/dx from the repo code plus autograd."""
    x_code = x.reshape(1, -1, 1, 1).detach().clone().requires_grad_(True)

    _, w_code_4d = fast_forward_matrix_linear_weights(
        model=model,
        x_bcos=x_code,
        target_class=0,
        create_graph=True,
    )
    w_code = w_code_4d.reshape(-1)

    jac_rows = []
    for w_i in w_code:
        grad_i = torch.autograd.grad(
            outputs=w_i,
            inputs=x_code,
            retain_graph=True,
            create_graph=False,
        )[0]
        jac_rows.append(grad_i.reshape(-1))

    return w_code.detach(), torch.stack(jac_rows).detach()


def print_tensor(name: str, value: Tensor) -> None:
    print(f"{name}:")
    print(value.detach().cpu())
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare hand-derived B-cos W(x), dW/dx formulas with repo autograd code.",
    )
    parser.add_argument("--atol", type=float, default=1e-10, help="Absolute tolerance for allclose.")
    parser.add_argument("--rtol", type=float, default=1e-8, help="Relative tolerance for allclose.")
    args = parser.parse_args()

    torch.set_printoptions(precision=10, sci_mode=False)
    dtype = torch.float64
    eps = 1e-6

    # Fixed values chosen so a^T x is not close to 0.
    raw_weight = torch.tensor([[[[0.8]], [[-0.4]], [[1.2]], [[0.3]]]], dtype=dtype)
    x = torch.tensor([0.5, -0.2, 1.0, 0.7], dtype=dtype)

    model = TinyBcosModel(raw_weight).eval()
    a = normalized_weight(raw_weight.reshape(-1))

    w_manual = manual_w(a, x, eps)
    d_w_dx_manual = manual_d_w_dx(a, x, eps)

    w_code, d_w_dx_code = code_w_and_d_w_dx(model, x)

    w_error = (w_manual - w_code).abs().max().item()
    jac_error = (d_w_dx_manual - d_w_dx_code).abs().max().item()

    print("=== Input ===")
    print_tensor("x", x)
    print_tensor("normalized linear weight a", a)

    print("=== W(x) comparison ===")
    print_tensor("manual W(x)", w_manual)
    print_tensor("code W(x)", w_code)
    print(f"max |manual W - code W| = {w_error:.12g}")
    print(f"allclose = {torch.allclose(w_manual, w_code, atol=args.atol, rtol=args.rtol)}")
    print()

    print("=== dW/dx comparison ===")
    print_tensor("manual dW/dx", d_w_dx_manual)
    print_tensor("code/autograd dW/dx", d_w_dx_code)
    print(f"max |manual dW/dx - code dW/dx| = {jac_error:.12g}")
    print(f"allclose = {torch.allclose(d_w_dx_manual, d_w_dx_code, atol=args.atol, rtol=args.rtol)}")
    print()

    if not torch.allclose(w_manual, w_code, atol=args.atol, rtol=args.rtol):
        raise SystemExit("W(x) formula does not match repo code.")
    if not torch.allclose(d_w_dx_manual, d_w_dx_code, atol=args.atol, rtol=args.rtol):
        raise SystemExit("dW/dx formula does not match repo code.")

    print("PASS: hand formulas match the repo code for this B-cos unit.")


if __name__ == "__main__":
    main()

