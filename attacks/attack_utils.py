from __future__ import annotations

import math

import torch
from torch import Tensor


def canonicalize_norm(norm: str) -> str:
    key = norm.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    aliases = {
        "l0": "l0",
        "0": "l0",
        "l2": "l2",
        "2": "l2",
        "linf": "linf",
        "inf": "linf",
        "infinity": "linf",
        "linfinity": "linf",
        "l∞": "linf",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported norm '{norm}'. Use one of: l0, l2, linf.") from exc


def _ensure_batch_dim(perturbation: Tensor) -> tuple[Tensor, bool]:
    if perturbation.dim() == 3:
        return perturbation.unsqueeze(0), True
    return perturbation, False


def _resolve_l0_budget(epsilon: float, total_dims: int) -> int:
    if epsilon <= 0:
        return 0
    if epsilon <= 1:
        return int(math.floor(total_dims * epsilon))
    return int(math.floor(epsilon))


def project_perturbation(perturbation: Tensor, epsilon: float, norm: str) -> Tensor:
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")

    norm = canonicalize_norm(norm)
    batched, squeeze_batch = _ensure_batch_dim(perturbation)

    if norm == "linf":
        projected = batched.clamp(-epsilon, epsilon)
    else:
        flat = batched.reshape(batched.shape[0], -1)
        if norm == "l2":
            norms = flat.norm(p=2, dim=1, keepdim=True)
            scale = torch.clamp(epsilon / (norms + 1e-12), max=1.0)
            projected = (flat * scale).view_as(batched)
        elif norm == "l0":
            keep = _resolve_l0_budget(epsilon, flat.shape[1])
            if keep <= 0:
                projected = torch.zeros_like(batched)
            elif keep >= flat.shape[1]:
                projected = batched
            else:
                _, topk_idx = flat.abs().topk(keep, dim=1, largest=True, sorted=False)
                mask = torch.zeros_like(flat, dtype=torch.bool)
                mask.scatter_(1, topk_idx, True)
                projected = torch.where(mask, flat, torch.zeros_like(flat)).view_as(batched)
        else:
            raise AssertionError(f"Unhandled norm '{norm}'.")

    if squeeze_batch:
        return projected.squeeze(0)
    return projected
