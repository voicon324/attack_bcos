"""
Explain-guided Evolution Strategies patch attack for B-cos models.

This variant uses `model.explain()` only to choose the patch location.
It keeps the original contribution-map position and adds a second
position from the contribution map of the current runner-up class,
then scores each candidate patch by the better of the two placements.
After that, it runs the same fixed-position ES attack pattern as
`bcos_es_patch_sweep.py`, directly optimizing the full-resolution patch
perturbation under an l0/l2/linf budget.
"""

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision import models as tv_models, transforms

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

IMAGENET_SL = 224
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


@dataclass(frozen=True)
class OriginalAttackPreprocess:
    image_transform: Any
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


ORIGINAL_ATTACK_PREPROCESS: Optional[OriginalAttackPreprocess] = None


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


def model_forward_rgb(
    model: torch.nn.Module,
    x_rgb: Tensor,
    channels_last: bool,
) -> Tensor:
    x_bcos = to_bcos_input(x_rgb)
    x_bcos = maybe_channels_last(x_bcos, channels_last)
    return model(x_bcos)


def normalize_for_original_model(x_rgb: Tensor) -> Tensor:
    if ORIGINAL_ATTACK_PREPROCESS is None:
        raise RuntimeError("Original attack preprocess is not configured.")
    mean = x_rgb.new_tensor(ORIGINAL_ATTACK_PREPROCESS.mean).view(1, 3, 1, 1)
    std = x_rgb.new_tensor(ORIGINAL_ATTACK_PREPROCESS.std).view(1, 3, 1, 1)
    return (x_rgb - mean) / std


def model_forward_original(
    model: torch.nn.Module,
    x_rgb: Tensor,
    channels_last: bool,
) -> Tensor:
    x_norm = normalize_for_original_model(x_rgb)
    x_norm = maybe_channels_last(x_norm, channels_last)
    return model(x_norm)


def model_forward_attack(
    model: torch.nn.Module,
    x_rgb: Tensor,
    channels_last: bool,
    attack_original_model: bool,
) -> Tensor:
    if attack_original_model:
        return model_forward_original(
            model,
            x_rgb,
            channels_last=channels_last,
        )
    return model_forward_rgb(
        model,
        x_rgb,
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
    print(
        f"  warning      : {model_name} does not provide IMAGENET1K_V1; "
        f"falling back to DEFAULT"
    )
    return weights_enum.DEFAULT


def build_original_attack_preprocess(weights: Any) -> OriginalAttackPreprocess:
    weight_transforms = weights.transforms()
    return OriginalAttackPreprocess(
        image_transform=transforms.Compose(
            [
                transforms.Resize(IMAGENET_SL),
                transforms.CenterCrop(IMAGENET_SL),
                transforms.ToTensor(),
            ]
        ),
        mean=tuple(float(v) for v in weight_transforms.mean),
        std=tuple(float(v) for v in weight_transforms.std),
    )


def load_original_attack_model(
    model_name: str,
    device: torch.device,
    channels_last: bool,
) -> Tuple[torch.nn.Module, str, OriginalAttackPreprocess]:
    torchvision_name = resolve_torchvision_model_name(model_name)
    weights = resolve_original_model_weights(torchvision_name)
    preprocess = build_original_attack_preprocess(weights)
    model = tv_models.get_model(torchvision_name, weights=weights).to(device).eval()
    if channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model, torchvision_name, preprocess


def load_attack_rgb_image(
    image_path: Optional[str],
    device: torch.device,
    attack_original_model: bool,
) -> Tuple[Tensor, str]:
    if not attack_original_model:
        return load_rgb_image(image_path, device=device)
    if ORIGINAL_ATTACK_PREPROCESS is None:
        raise RuntimeError("Original attack preprocess is not configured.")
    if image_path is None:
        raise ValueError("An image path is required when attacking the original model.")

    with Image.open(image_path) as img:
        x_rgb = ORIGINAL_ATTACK_PREPROCESS.image_transform(img.convert("RGB"))
    return x_rgb.unsqueeze(0).to(device), image_path


def compute_cw_scores(outputs: Tensor, target_classes: Tensor) -> Tensor:
    if target_classes.dim() == 0:
        target_classes = target_classes.unsqueeze(0)
    target_idx = target_classes.view(-1, 1)
    target_logits = outputs.gather(1, target_idx).squeeze(1)
    out_clone = outputs.clone()
    out_clone.scatter_(1, target_idx, -float("inf"))
    max_other = out_clone.max(dim=1).values
    return max_other - target_logits


def resolve_runner_up_classes(outputs: Tensor, target_classes: Tensor) -> Tensor:
    if target_classes.dim() == 0:
        target_classes = target_classes.unsqueeze(0)
    target_idx = target_classes.view(-1, 1)
    out_clone = outputs.clone()
    out_clone.scatter_(1, target_idx, -float("inf"))
    return out_clone.argmax(dim=1)


def resolve_total_candidate_batch_size(
    x_rgb: Tensor,
    score_batch_size: int,
    attack_original_model: bool,
) -> int:
    if score_batch_size > 0:
        return score_batch_size
    if x_rgb.device.type != "cuda":
        return 64
    if attack_original_model:
        # Torchvision models over large image batches are much more memory-hungry.
        return 1024
    return 4096


def squeeze_contribution_map(contribution_map: Tensor) -> Tensor:
    cmap = contribution_map
    if cmap.dim() == 4:
        cmap = cmap.squeeze(0)
    if cmap.dim() == 3:
        cmap = cmap.squeeze(0)
    return cmap


def find_best_patch_position_from_importance_map(
    importance_map: Tensor,
    patch_size: int,
) -> Tuple[int, int]:
    attr = importance_map.detach().abs().cpu().numpy().astype(np.float64)
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


def patch_window_sums(map_2d: np.ndarray, patch_size: int) -> np.ndarray:
    h, w = map_2d.shape
    s = min(patch_size, h, w)
    prefix = np.zeros((h + 1, w + 1), dtype=np.float64)
    prefix[1:, 1:] = np.cumsum(np.cumsum(map_2d, axis=0), axis=1)
    return (
        prefix[s:, s:]
        - prefix[:-s, s:]
        - prefix[s:, :-s]
        + prefix[:-s, :-s]
    )


def find_best_patch_position_from_contribution_margin(
    primary_contribution_map: Tensor,
    secondary_contribution_map: Tensor,
    patch_size: int,
) -> Tuple[int, int]:
    primary = squeeze_contribution_map(primary_contribution_map).detach().cpu().numpy().astype(np.float64)
    secondary = squeeze_contribution_map(secondary_contribution_map).detach().cpu().numpy().astype(np.float64)
    if primary.shape != secondary.shape:
        raise ValueError(
            "Primary and secondary contribution maps must have the same shape, "
            f"got {primary.shape} and {secondary.shape}."
        )

    margin_sums = patch_window_sums(primary, patch_size) - patch_window_sums(secondary, patch_size)
    best_flat = int(np.argmax(margin_sums))
    best_y, best_x = np.unravel_index(best_flat, margin_sums.shape)
    return int(best_y), int(best_x)


def find_best_patch_position_from_contribution_map(
    contribution_map: Tensor,
    patch_size: int,
) -> Tuple[int, int]:
    contribution = squeeze_contribution_map(contribution_map).detach().cpu().numpy().astype(np.float64)
    contribution_sums = patch_window_sums(contribution, patch_size)
    best_flat = int(np.argmax(contribution_sums))
    best_y, best_x = np.unravel_index(best_flat, contribution_sums.shape)
    return int(best_y), int(best_x)


def dynamic_linear_weights_to_importance_map(dynamic_linear_weights: Tensor) -> Tensor:
    if dynamic_linear_weights.dim() == 3:
        dynamic_linear_weights = dynamic_linear_weights.unsqueeze(0)
    if dynamic_linear_weights.dim() != 4 or dynamic_linear_weights.shape[1] != 6:
        raise ValueError(
            "Expected dynamic_linear_weights with shape [B, 6, H, W] from model.explain()."
        )

    importance = dynamic_linear_weights.norm(p=2, dim=1)
    if importance.shape[0] != 1:
        raise ValueError(f"Expected one dynamic importance map, got batch size {importance.shape[0]}.")
    return importance.squeeze(0)


def find_best_patch_position_from_dynamic_map(
    dynamic_linear_weights: Tensor,
    patch_size: int,
) -> Tuple[int, int]:
    dynamic_map = dynamic_linear_weights_to_importance_map(dynamic_linear_weights)
    dynamic_sums = patch_window_sums(dynamic_map.detach().cpu().numpy().astype(np.float64), patch_size)
    best_flat = int(np.argmax(dynamic_sums))
    best_y, best_x = np.unravel_index(best_flat, dynamic_sums.shape)
    return int(best_y), int(best_x)


def find_best_patch_position_from_dynamic_margin(
    primary_dynamic_linear_weights: Tensor,
    secondary_dynamic_linear_weights: Tensor,
    patch_size: int,
) -> Tuple[int, int]:
    primary = dynamic_linear_weights_to_importance_map(primary_dynamic_linear_weights)
    secondary = dynamic_linear_weights_to_importance_map(secondary_dynamic_linear_weights)
    primary_np = primary.detach().cpu().numpy().astype(np.float64)
    secondary_np = secondary.detach().cpu().numpy().astype(np.float64)
    if primary_np.shape != secondary_np.shape:
        raise ValueError(
            "Primary and secondary dynamic maps must have the same shape, "
            f"got {primary_np.shape} and {secondary_np.shape}."
        )

    margin_sums = patch_window_sums(primary_np, patch_size) - patch_window_sums(secondary_np, patch_size)
    best_flat = int(np.argmax(margin_sums))
    best_y, best_x = np.unravel_index(best_flat, margin_sums.shape)
    return int(best_y), int(best_x)


def find_best_patch_position(
    contribution_map: Tensor,
    patch_size: int,
) -> Tuple[int, int]:
    cmap = squeeze_contribution_map(contribution_map)
    return find_best_patch_position_from_importance_map(cmap, patch_size)


def find_random_patch_position(
    contribution_map: Tensor,
    patch_size: int,
) -> Tuple[int, int]:
    cmap = squeeze_contribution_map(contribution_map)
    h, w = cmap.shape
    s = min(patch_size, h, w)
    return int(np.random.randint(0, h - s + 1)), int(np.random.randint(0, w - s + 1))


def resolve_patch_positions(
    primary_contribution_map: Tensor,
    secondary_contribution_map: Tensor,
    primary_dynamic_linear_weights: Tensor,
    secondary_dynamic_linear_weights: Tensor,
    patch_size: int,
    pos_y: int,
    pos_x: int,
    use_explain_position: bool,
    position_rule: str,
) -> List[Tuple[int, int]]:
    if not use_explain_position:
        primary_pos = (max(0, pos_y), max(0, pos_x))
    elif position_rule == "margin":
        primary_pos = find_best_patch_position_from_contribution_margin(
            primary_contribution_map,
            secondary_contribution_map,
            patch_size,
        )
    elif position_rule == "top1":
        primary_pos = find_best_patch_position_from_contribution_map(primary_contribution_map, patch_size)
    elif position_rule == "dynamic-margin":
        primary_pos = find_best_patch_position_from_dynamic_margin(
            primary_dynamic_linear_weights,
            secondary_dynamic_linear_weights,
            patch_size,
        )
    elif position_rule == "dynamic":
        primary_pos = find_best_patch_position_from_dynamic_map(primary_dynamic_linear_weights, patch_size)
    elif position_rule == "random":
        primary_pos = find_random_patch_position(primary_contribution_map, patch_size)
    else:
        raise ValueError(f"Unsupported position rule: {position_rule}")
    # secondary_pos = find_best_patch_position(
    #     secondary_contribution_map,
    #     patch_size,
    # )
    # return [primary_pos, secondary_pos]
    return [primary_pos]


class ExplainGuidedPatchES:
    def __init__(
        self,
        patch_size: int,
        population_size: int,
        elite_fraction: float,
        sigma: float,
        sigma_decay: float,
        sigma_min: float,
        learning_rate: float,
        epsilon: float,
        norm: str,
        device: torch.device,
    ):
        self.param_shape = (1, 3, patch_size, patch_size)
        self.population_size = population_size
        self.elite_size = max(1, int(population_size * elite_fraction))
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min
        self.lr = learning_rate
        self.epsilon = epsilon
        self.norm = canonicalize_norm(norm)
        self.device = device
        self.perturbation = torch.zeros(self.param_shape, device=device)

    def ask(self) -> Tensor:
        noise = torch.randn(self.population_size, *self.param_shape, device=self.device)
        candidates = self.perturbation.unsqueeze(0) + self.sigma * noise
        return project_perturbation(candidates, self.epsilon, self.norm)

    def tell(self, candidates: Tensor, scores: Tensor) -> None:
        _, elite_idx = scores.topk(self.elite_size)
        elite = candidates[elite_idx]
        elite_scores = scores[elite_idx]
        elite_scores = (elite_scores - elite_scores.mean()) / (elite_scores.std(unbiased=False) + 1e-8)
        weights = F.softmax(elite_scores, dim=0)

        w = weights.clone()
        for _ in range(len(self.param_shape)):
            w = w.unsqueeze(-1)

        new_patch = (w * elite).sum(dim=0)
        self.perturbation = (1.0 - self.lr) * self.perturbation + self.lr * new_patch
        self.perturbation = project_perturbation(self.perturbation, self.epsilon, self.norm)
        self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)

    def get_patch(self) -> Tensor:
        return self.perturbation.clone()


def compute_scores_fixed_multi_position(
    model: torch.nn.Module,
    x_rgb: Tensor,
    candidate_patches: Tensor,
    patch_positions: List[Tuple[int, int]],
    target_class: int,
    score_batch_size: int,
    channels_last: bool,
    attack_original_model: bool,
) -> Tuple[Tensor, Tensor]:
    n = candidate_patches.shape[0]
    position_count = len(patch_positions)
    total_eval_batch = resolve_total_candidate_batch_size(
        x_rgb,
        score_batch_size=score_batch_size,
        attack_original_model=attack_original_model,
    )
    batch_size = min(n, max(1, total_eval_batch // max(position_count, 1)))
    scores_list: List[Tensor] = []
    pos_idx_list: List[Tensor] = []
    patch_size = candidate_patches.shape[-1]

    with torch.inference_mode():
        for i in range(0, n, batch_size):
            patch_batch = candidate_patches[i:i + batch_size]
            batch = patch_batch.shape[0]

            x_rep = x_rgb.expand(batch * position_count, -1, -1, -1).clone()
            x_rep = maybe_channels_last(x_rep, channels_last)
            patch_vals = patch_batch.squeeze(1)
            for pos_idx, (pos_y, pos_x) in enumerate(patch_positions):
                lo = pos_idx * batch
                hi = lo + batch
                x_rep[lo:hi, :, pos_y:pos_y + patch_size, pos_x:pos_x + patch_size] = torch.clamp(
                    x_rep[lo:hi, :, pos_y:pos_y + patch_size, pos_x:pos_x + patch_size] + patch_vals,
                    0.0,
                    1.0,
                )

            outputs = model_forward_attack(
                model,
                x_rep,
                channels_last=channels_last,
                attack_original_model=attack_original_model,
            )
            per_position_scores = compute_cw_scores(
                outputs,
                torch.full((outputs.shape[0],), target_class, device=outputs.device, dtype=torch.long),
            ).view(position_count, batch)
            best_scores, best_pos_idx = per_position_scores.max(dim=0)
            scores_list.append(best_scores.float())
            pos_idx_list.append(best_pos_idx.to(dtype=torch.long))

    return torch.cat(scores_list), torch.cat(pos_idx_list)


def select_best_position_for_patch(
    model: torch.nn.Module,
    x_rgb: Tensor,
    patch: Tensor,
    patch_positions: List[Tuple[int, int]],
    target_class: int,
    score_batch_size: int,
    channels_last: bool,
    attack_original_model: bool,
) -> Tuple[int, int]:
    _, best_pos_idx = compute_scores_fixed_multi_position(
        model,
        x_rgb,
        patch.unsqueeze(0),
        patch_positions,
        target_class,
        score_batch_size=score_batch_size,
        channels_last=channels_last,
        attack_original_model=attack_original_model,
    )
    best_pos = patch_positions[int(best_pos_idx[0].item())]
    return best_pos[0], best_pos[1]


def load_rgb_image_batch(
    image_items: List[Tuple[Optional[int], str]],
    device: torch.device,
    channels_last: bool,
    attack_original_model: bool,
) -> Tuple[Tensor, List[str]]:
    batch_tensors: List[Tensor] = []
    resolved_paths: List[str] = []
    for _, image_path in image_items:
        x_rgb, resolved_image_path = load_attack_rgb_image(
            image_path,
            device=device,
            attack_original_model=attack_original_model,
        )
        batch_tensors.append(x_rgb)
        resolved_paths.append(resolved_image_path)
    x_rgb_batch = torch.cat(batch_tensors, dim=0)
    x_rgb_batch = maybe_channels_last(x_rgb_batch, channels_last)
    return x_rgb_batch, resolved_paths


def apply_patches_batch(
    x_rgb: Tensor,
    patches: Tensor,
    pos_y: List[int],
    pos_x: List[int],
) -> Tensor:
    x_pert = x_rgb.clone()
    patch_vals = patches.squeeze(1)
    patch_size = patch_vals.shape[-1]
    for image_idx in range(x_rgb.shape[0]):
        y = int(pos_y[image_idx])
        x = int(pos_x[image_idx])
        x_pert[image_idx:image_idx + 1, :, y:y + patch_size, x:x + patch_size] = torch.clamp(
            x_pert[image_idx:image_idx + 1, :, y:y + patch_size, x:x + patch_size] + patch_vals[image_idx:image_idx + 1],
            0.0,
            1.0,
        )
    return x_pert


def normalize_patch_for_visualization(patch: Tensor) -> Tensor:
    patch_4d = patch
    while patch_4d.dim() > 4 and patch_4d.shape[0] == 1:
        patch_4d = patch_4d.squeeze(0)
    if patch_4d.dim() == 3:
        patch_4d = patch_4d.unsqueeze(0)
    if patch_4d.dim() != 4 or patch_4d.shape[0] != 1 or patch_4d.shape[1] != 3:
        raise ValueError(
            f"Expected patch shaped like (1, 3, H, W) after squeezing singleton dims, got {tuple(patch.shape)}"
        )
    return patch_4d.contiguous()


class BatchedExplainGuidedPatchES:
    def __init__(
        self,
        batch_size: int,
        patch_size: int,
        population_size: int,
        elite_fraction: float,
        sigma: float,
        sigma_decay: float,
        sigma_min: float,
        learning_rate: float,
        epsilon: float,
        norm: str,
        device: torch.device,
    ):
        self.patch_shape = (1, 3, patch_size, patch_size)
        self.batch_size = batch_size
        self.population_size = population_size
        self.elite_size = max(1, int(population_size * elite_fraction))
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min
        self.lr = learning_rate
        self.epsilon = epsilon
        self.norm = canonicalize_norm(norm)
        self.device = device
        self.perturbation = torch.zeros((batch_size, *self.patch_shape), device=device)

    def ask(self) -> Tensor:
        noise = torch.randn(
            self.batch_size,
            self.population_size,
            *self.patch_shape,
            device=self.device,
        )
        candidates = self.perturbation.unsqueeze(1) + self.sigma * noise
        flat_candidates = candidates.reshape(-1, *self.patch_shape)
        flat_candidates = project_perturbation(flat_candidates, self.epsilon, self.norm)
        return flat_candidates.view_as(candidates)

    def tell(self, candidates: Tensor, scores: Tensor) -> None:
        elite_idx = scores.topk(self.elite_size, dim=1).indices
        gather_idx = elite_idx.view(self.batch_size, self.elite_size, 1, 1, 1, 1).expand(
            -1,
            -1,
            *self.patch_shape,
        )
        elite = candidates.gather(1, gather_idx)
        elite_scores = scores.gather(1, elite_idx)
        elite_scores = (
            elite_scores - elite_scores.mean(dim=1, keepdim=True)
        ) / (elite_scores.std(dim=1, keepdim=True, unbiased=False) + 1e-8)
        weights = F.softmax(elite_scores, dim=1)
        new_patch = (weights.view(self.batch_size, self.elite_size, 1, 1, 1, 1) * elite).sum(dim=1)
        self.perturbation = (1.0 - self.lr) * self.perturbation + self.lr * new_patch
        self.perturbation = project_perturbation(self.perturbation, self.epsilon, self.norm)
        self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)

    def get_patches(self) -> Tensor:
        return self.perturbation.clone()


def compute_scores_fixed_batched_multi_position(
    model: torch.nn.Module,
    x_rgb: Tensor,
    candidate_patches: Tensor,
    patch_positions: List[List[Tuple[int, int]]],
    target_classes: Tensor,
    score_batch_size: int,
    channels_last: bool,
    attack_original_model: bool,
) -> Tuple[Tensor, Tensor]:
    batch_size, population_size = candidate_patches.shape[:2]
    position_count = len(patch_positions[0]) if patch_positions else 1
    total_eval_batch = resolve_total_candidate_batch_size(
        x_rgb,
        score_batch_size=score_batch_size,
        attack_original_model=attack_original_model,
    )
    pop_chunk = max(
        1,
        min(population_size, total_eval_batch // max(batch_size * position_count, 1)),
    )

    patch_size = candidate_patches.shape[-1]
    scores_list: List[Tensor] = []
    pos_idx_list: List[Tensor] = []

    with torch.inference_mode():
        for start in range(0, population_size, pop_chunk):
            patch_chunk = candidate_patches[:, start:start + pop_chunk]
            chunk_size = patch_chunk.shape[1]

            x_rep = x_rgb.unsqueeze(1).unsqueeze(2).expand(-1, position_count, chunk_size, -1, -1, -1).clone()
            x_rep = x_rep.reshape(batch_size * position_count * chunk_size, *x_rgb.shape[1:])
            x_rep = maybe_channels_last(x_rep, channels_last)

            patch_vals = patch_chunk.squeeze(2)
            for image_idx in range(batch_size):
                patch_vals_image = patch_vals[image_idx]
                for pos_idx, (y, x) in enumerate(patch_positions[image_idx]):
                    lo = (image_idx * position_count + pos_idx) * chunk_size
                    hi = lo + chunk_size
                    x_rep[lo:hi, :, y:y + patch_size, x:x + patch_size] = torch.clamp(
                        x_rep[lo:hi, :, y:y + patch_size, x:x + patch_size] + patch_vals_image,
                        0.0,
                        1.0,
                    )

            outputs = model_forward_attack(
                model,
                x_rep,
                channels_last=channels_last,
                attack_original_model=attack_original_model,
            ).view(batch_size, position_count, chunk_size, -1)
            flat_outputs = outputs.reshape(batch_size * position_count * chunk_size, -1)
            flat_targets = target_classes.unsqueeze(1).unsqueeze(2).expand(-1, position_count, chunk_size).reshape(-1)
            flat_scores = compute_cw_scores(flat_outputs, flat_targets)
            per_position_scores = flat_scores.view(batch_size, position_count, chunk_size)
            best_scores, best_pos_idx = per_position_scores.max(dim=1)
            scores_list.append(best_scores.float())
            pos_idx_list.append(best_pos_idx.to(dtype=torch.long))

    return torch.cat(scores_list, dim=1), torch.cat(pos_idx_list, dim=1)


def run_explain_guided_attack(
    guide_model: torch.nn.Module,
    attack_model: torch.nn.Module,
    x_rgb: Tensor,
    target_class: int,
    secondary_class: int,
    patch_size: int,
    generations: int,
    population_size: int,
    elite_fraction: float,
    sigma: float,
    sigma_decay: float,
    sigma_min: float,
    learning_rate: float,
    epsilon: float,
    norm: str,
    pos_y: int,
    pos_x: int,
    use_explain_position: bool,
    position_rule: str,
    score_batch_size: int,
    channels_last: bool,
    attack_original_model: bool,
    verbose: bool,
) -> Tuple[Tensor, int, int, List[Tuple[int, float, float, float]]]:
    primary_guidance = extract_attribution(guide_model, to_bcos_input(x_rgb), target_class=target_class)
    secondary_guidance = extract_attribution(guide_model, to_bcos_input(x_rgb), target_class=secondary_class)
    patch_positions = resolve_patch_positions(
        primary_guidance["contribution_map"],
        secondary_guidance["contribution_map"],
        primary_guidance["dynamic_linear_weights"],
        secondary_guidance["dynamic_linear_weights"],
        patch_size,
        pos_y=pos_y,
        pos_x=pos_x,
        use_explain_position=use_explain_position,
        position_rule=position_rule,
    )

    es = ExplainGuidedPatchES(
        patch_size=patch_size,
        population_size=population_size,
        elite_fraction=elite_fraction,
        sigma=sigma,
        sigma_decay=sigma_decay,
        sigma_min=sigma_min,
        learning_rate=learning_rate,
        epsilon=epsilon,
        norm=norm,
        device=x_rgb.device,
    )

    history: List[Tuple[int, float, float, float]] = []
    best_patch_overall: Optional[Tensor] = None
    best_score_overall = -float("inf")
    best_pos_overall = patch_positions[0]

    for gen in range(generations):
        candidate_patches = es.ask()
        scores, best_pos_idx = compute_scores_fixed_multi_position(
            attack_model,
            x_rgb,
            candidate_patches,
            patch_positions,
            target_class,
            score_batch_size=score_batch_size,
            channels_last=channels_last,
            attack_original_model=attack_original_model,
        )
        es.tell(candidate_patches, scores)

        best_idx = scores.argmax().item()
        best_patch = candidate_patches[best_idx]
        best_pos = patch_positions[int(best_pos_idx[best_idx].item())]
        best_x = apply_patch(x_rgb, best_patch, best_pos[0], best_pos[1])
        with torch.inference_mode():
            out_best = model_forward_attack(
                attack_model,
                best_x,
                channels_last=channels_last,
                attack_original_model=attack_original_model,
            )
            best_logit = out_best[0, target_class].item()

        best_score = scores[best_idx].item()
        if best_score > best_score_overall:
            best_score_overall = best_score
            best_patch_overall = best_patch.clone()
            best_pos_overall = best_pos
        history.append((gen, best_score, best_logit, es.sigma))
        if verbose:
            print(
                f"  Gen {gen + 1:3d}/{generations} | "
                f"Target Logit: {best_logit:7.3f} | "
                f"Score: {best_score:7.3f} | sigma: {es.sigma:.4f}",
                flush=True,
            )

    final_patch = es.get_patch()
    if best_patch_overall is not None:
        final_patch = best_patch_overall.clone()
    else:
        best_pos_overall = select_best_position_for_patch(
            attack_model,
            x_rgb,
            final_patch,
            patch_positions,
            target_class,
            score_batch_size=score_batch_size,
            channels_last=channels_last,
            attack_original_model=attack_original_model,
        )
    return final_patch, best_pos_overall[0], best_pos_overall[1], history


def run_explain_guided_attack_batch(
    guide_model: torch.nn.Module,
    attack_model: torch.nn.Module,
    x_rgb: Tensor,
    target_classes: Tensor,
    secondary_classes: Tensor,
    patch_size: int,
    generations: int,
    population_size: int,
    elite_fraction: float,
    sigma: float,
    sigma_decay: float,
    sigma_min: float,
    learning_rate: float,
    epsilon: float,
    norm: str,
    pos_y: int,
    pos_x: int,
    use_explain_position: bool,
    position_rule: str,
    score_batch_size: int,
    channels_last: bool,
    attack_original_model: bool,
    verbose: bool,
) -> Tuple[Tensor, List[int], List[int], List[List[Tuple[int, float, float, float]]]]:
    batch_size = x_rgb.shape[0]
    patch_positions: List[List[Tuple[int, int]]] = []
    for image_idx in range(batch_size):
        target_class = int(target_classes[image_idx].item())
        secondary_class = int(secondary_classes[image_idx].item())
        primary_guidance = extract_attribution(
            guide_model,
            to_bcos_input(x_rgb[image_idx:image_idx + 1]),
            target_class=target_class,
        )
        secondary_guidance = extract_attribution(
            guide_model,
            to_bcos_input(x_rgb[image_idx:image_idx + 1]),
            target_class=secondary_class,
        )
        patch_positions.append(
            resolve_patch_positions(
                primary_guidance["contribution_map"],
                secondary_guidance["contribution_map"],
                primary_guidance["dynamic_linear_weights"],
                secondary_guidance["dynamic_linear_weights"],
                patch_size,
                pos_y=pos_y,
                pos_x=pos_x,
                use_explain_position=use_explain_position,
                position_rule=position_rule,
            )
        )

    es = BatchedExplainGuidedPatchES(
        batch_size=batch_size,
        patch_size=patch_size,
        population_size=population_size,
        elite_fraction=elite_fraction,
        sigma=sigma,
        sigma_decay=sigma_decay,
        sigma_min=sigma_min,
        learning_rate=learning_rate,
        epsilon=epsilon,
        norm=norm,
        device=x_rgb.device,
    )

    histories: List[List[Tuple[int, float, float, float]]] = [[] for _ in range(batch_size)]
    best_score_overall = torch.full((batch_size,), -float("inf"), device=x_rgb.device)
    best_patches_overall = es.get_patches()
    best_pos_y_overall = [positions[0][0] for positions in patch_positions]
    best_pos_x_overall = [positions[0][1] for positions in patch_positions]

    for gen in range(generations):
        candidate_patches = es.ask()
        scores, best_pos_idx = compute_scores_fixed_batched_multi_position(
            attack_model,
            x_rgb,
            candidate_patches,
            patch_positions,
            target_classes,
            score_batch_size=score_batch_size,
            channels_last=channels_last,
            attack_original_model=attack_original_model,
        )
        es.tell(candidate_patches, scores)

        best_scores, best_idx = scores.max(dim=1)
        gather_idx = best_idx.view(batch_size, 1, 1, 1, 1, 1).expand(
            -1,
            1,
            *candidate_patches.shape[2:],
        )
        best_patches = candidate_patches.gather(1, gather_idx).squeeze(1)
        best_pos_choice = best_pos_idx.gather(1, best_idx.unsqueeze(1)).squeeze(1)
        best_pos_y = [
            patch_positions[image_idx][int(best_pos_choice[image_idx].item())][0]
            for image_idx in range(batch_size)
        ]
        best_pos_x = [
            patch_positions[image_idx][int(best_pos_choice[image_idx].item())][1]
            for image_idx in range(batch_size)
        ]
        best_x = apply_patches_batch(x_rgb, best_patches, best_pos_y, best_pos_x)
        with torch.inference_mode():
            out_best = model_forward_attack(
                attack_model,
                best_x,
                channels_last=channels_last,
                attack_original_model=attack_original_model,
            )
            best_logits = out_best.gather(1, target_classes.unsqueeze(1)).squeeze(1)

        improved = best_scores > best_score_overall
        best_score_overall = torch.where(improved, best_scores, best_score_overall)
        best_patches_overall[improved] = best_patches[improved]
        for image_idx in range(batch_size):
            if bool(improved[image_idx].item()):
                best_pos_y_overall[image_idx] = best_pos_y[image_idx]
                best_pos_x_overall[image_idx] = best_pos_x[image_idx]

        for image_idx in range(batch_size):
            histories[image_idx].append(
                (
                    gen,
                    float(best_scores[image_idx].item()),
                    float(best_logits[image_idx].item()),
                    es.sigma,
                )
            )

        if verbose:
            if batch_size == 1:
                print(
                    f"  Gen {gen + 1:3d}/{generations} | "
                    f"Target Logit: {best_logits[0].item():7.3f} | "
                    f"Score: {best_scores[0].item():7.3f} | sigma: {es.sigma:.4f}",
                    flush=True,
                )
            else:
                print(
                    f"  Gen {gen + 1:3d}/{generations} | "
                    f"Mean Target Logit: {best_logits.mean().item():7.3f} | "
                    f"Best Target Logit: {best_logits.min().item():7.3f} | "
                    f"Mean Score: {best_scores.mean().item():7.3f} | sigma: {es.sigma:.4f}",
                    flush=True,
                )

    return best_patches_overall, best_pos_y_overall, best_pos_x_overall, histories


def build_image_output_dir(output_root: Path, image_path: str, image_index: Optional[int]) -> Path:
    image_file = Path(image_path)
    stem_parts: List[str] = []
    if image_index is not None:
        stem_parts.append(f"{image_index:03d}")
    if image_file.parent.name:
        stem_parts.append(image_file.parent.name)
    stem_parts.append(image_file.stem)
    safe_name = "_".join(part for part in stem_parts if part)
    return output_root / f"{safe_name}_explain_guided"


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


def resolve_plot_save_path(save_arg: str, out_dir: Path) -> str:
    if save_arg == "explain_guided_es_patch.png":
        return str(out_dir / "explain_guided_es_patch.png")
    save_path = Path(save_arg)
    if save_path.is_absolute():
        return str(save_path)
    return str(out_dir / save_path.name)


def process_single_image(
    guide_model: torch.nn.Module,
    attack_model: torch.nn.Module,
    image_path: str,
    image_index: Optional[int],
    output_root: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    x_rgb, resolved_image_path = load_attack_rgb_image(
        image_path,
        device=args.device_obj,
        attack_original_model=args.attack_original_model,
    )
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
            channels_last=args.channels_last,
            attack_original_model=args.attack_original_model,
        )
        target_class = out_orig.argmax(1).item()
        secondary_class = resolve_runner_up_classes(
            out_orig,
            torch.tensor([target_class], device=out_orig.device, dtype=torch.long),
        )[0].item()
        logit_orig = out_orig[0, target_class].item()

    print(f"  Original class: {target_class} (logit: {logit_orig:.3f})")
    print(f"  Runner-up class: {secondary_class}")

    print("\n▸ Running explain-guided ES optimization ...")
    final_patch, final_pos_y, final_pos_x, history = run_explain_guided_attack(
        guide_model=guide_model,
        attack_model=attack_model,
        x_rgb=x_rgb,
        target_class=target_class,
        secondary_class=secondary_class,
        patch_size=args.patch_size,
        generations=args.generations,
        population_size=args.population,
        elite_fraction=args.elite_fraction,
        sigma=args.sigma,
        sigma_decay=args.sigma_decay,
        sigma_min=args.sigma_min,
        learning_rate=args.lr,
        epsilon=args.epsilon,
        norm=args.norm,
        pos_y=0,
        pos_x=0,
        use_explain_position=True,
        position_rule=args.position_rule,
        score_batch_size=args.score_batch_size,
        channels_last=args.channels_last,
        attack_original_model=args.attack_original_model,
        verbose=args.verbose_generations,
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
            patch=normalize_patch_for_visualization(final_patch),
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
        "position_rule": args.position_rule,
        "norm": canonicalize_norm(args.norm),
        "epsilon": f"{args.epsilon:g}",
        "success": int(pred_class_pert != target_class),
    }


def process_image_batch(
    guide_model: torch.nn.Module,
    attack_model: torch.nn.Module,
    image_items: List[Tuple[Optional[int], str]],
    output_root: Path,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    x_rgb_batch, resolved_image_paths = load_rgb_image_batch(
        image_items,
        device=args.device_obj,
        channels_last=args.channels_last,
        attack_original_model=args.attack_original_model,
    )
    out_dirs = [
        build_image_output_dir(output_root, resolved_path, image_index)
        for (image_index, _), resolved_path in zip(image_items, resolved_image_paths)
    ]
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n▸ Loading image batch ... {len(image_items)} images")
    for batch_idx, ((_, _), resolved_path) in enumerate(zip(image_items, resolved_image_paths), start=1):
        print(f"  [{batch_idx:02d}/{len(image_items):02d}] {resolved_path}")

    if args.save_images:
        import torchvision.utils as vutils

        for image_idx, out_dir in enumerate(out_dirs):
            before_path = out_dir / "before.png"
            vutils.save_image(x_rgb_batch[image_idx:image_idx + 1], str(before_path))

    with torch.inference_mode():
        out_orig = model_forward_attack(
            attack_model,
            x_rgb_batch,
            channels_last=args.channels_last,
            attack_original_model=args.attack_original_model,
        )
        target_classes = out_orig.argmax(dim=1)
        secondary_classes = resolve_runner_up_classes(out_orig, target_classes)
        logit_orig = out_orig.gather(1, target_classes.unsqueeze(1)).squeeze(1)

    print("\n▸ Running batched explain-guided ES optimization ...")
    final_patches, final_pos_y, final_pos_x, histories = run_explain_guided_attack_batch(
        guide_model=guide_model,
        attack_model=attack_model,
        x_rgb=x_rgb_batch,
        target_classes=target_classes,
        secondary_classes=secondary_classes,
        patch_size=args.patch_size,
        generations=args.generations,
        population_size=args.population,
        elite_fraction=args.elite_fraction,
        sigma=args.sigma,
        sigma_decay=args.sigma_decay,
        sigma_min=args.sigma_min,
        learning_rate=args.lr,
        epsilon=args.epsilon,
        norm=args.norm,
        pos_y=0,
        pos_x=0,
        use_explain_position=True,
        position_rule=args.position_rule,
        score_batch_size=args.score_batch_size,
        channels_last=args.channels_last,
        attack_original_model=args.attack_original_model,
        verbose=args.verbose_generations,
    )

    final_patches = project_perturbation(final_patches, args.epsilon, args.norm)
    x_pert_batch = apply_patches_batch(x_rgb_batch, final_patches, final_pos_y, final_pos_x)
    with torch.inference_mode():
        out_pert = model_forward_attack(
            attack_model,
            x_pert_batch,
            channels_last=args.channels_last,
            attack_original_model=args.attack_original_model,
        )
        pred_classes_pert = out_pert.argmax(dim=1)
        logit_pert_tgt = out_pert.gather(1, target_classes.unsqueeze(1)).squeeze(1)
        logit_pert_pred = out_pert.gather(1, pred_classes_pert.unsqueeze(1)).squeeze(1)

    results: List[Dict[str, Any]] = []
    if args.save_images:
        import torchvision.utils as vutils

    for batch_idx, ((image_index, _), resolved_path, out_dir) in enumerate(
        zip(image_items, resolved_image_paths, out_dirs),
        start=1,
    ):
        target_class = int(target_classes[batch_idx - 1].item())
        pred_class_pert = int(pred_classes_pert[batch_idx - 1].item())
        orig_logit = float(logit_orig[batch_idx - 1].item())
        pert_logit_tgt = float(logit_pert_tgt[batch_idx - 1].item())
        pert_logit_pred = float(logit_pert_pred[batch_idx - 1].item())

        print(
            f"  [{batch_idx:02d}/{len(image_items):02d}] "
            f"class {target_class} | pos ({final_pos_y[batch_idx - 1]}, {final_pos_x[batch_idx - 1]}) | "
            f"logit {orig_logit:.3f} -> {pert_logit_tgt:.3f}"
        )

        if args.save_images:
            after_path = out_dir / "after.png"
            vutils.save_image(x_pert_batch[batch_idx - 1:batch_idx], str(after_path))

        if args.save_figure:
            attr_orig = extract_attribution(
                guide_model,
                to_bcos_input(x_rgb_batch[batch_idx - 1:batch_idx]),
                target_class=target_class,
            )
            attr_pert = extract_attribution(
                guide_model,
                to_bcos_input(x_pert_batch[batch_idx - 1:batch_idx]),
                target_class=target_class,
            )
            save_plot_path = resolve_plot_save_path(args.save, out_dir)
            visualize_patch_results(
                x_orig_rgb=x_rgb_batch[batch_idx - 1:batch_idx],
                x_pert_rgb=x_pert_batch[batch_idx - 1:batch_idx],
                patch=normalize_patch_for_visualization(final_patches[batch_idx - 1]),
                attr_orig=attr_orig,
                attr_pert=attr_pert,
                target_class=target_class,
                logit_orig=orig_logit,
                pred_class_pert=pred_class_pert,
                logit_pert_tgt=pert_logit_tgt,
                logit_pert_pred=pert_logit_pred,
                history=histories[batch_idx - 1],
                save_path=save_plot_path,
            )

        results.append(
            {
                "index": image_index,
                "image_path": resolved_path,
                "output_dir": str(out_dir),
                "target_class": target_class,
                "original_logit": orig_logit,
                "perturbed_class": pred_class_pert,
                "perturbed_target_logit": pert_logit_tgt,
                "perturbed_pred_logit": pert_logit_pred,
                "patch_pos_y": final_pos_y[batch_idx - 1],
                "patch_pos_x": final_pos_x[batch_idx - 1],
                "position_rule": args.position_rule,
                "norm": canonicalize_norm(args.norm),
                "epsilon": f"{args.epsilon:g}",
                "success": int(pred_class_pert != target_class),
            }
        )

    return results


def main() -> None:
    global ORIGINAL_ATTACK_PREPROCESS

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser(
        description="Explain-guided ES patch attack for B-cos models",
    )
    parser.add_argument("--image", type=str, default=None, help="Path to one input image")
    parser.add_argument("--images-csv", type=str, default=None, help="CSV containing an image_path column")
    parser.add_argument("--model", type=str, default="resnet50", help="B-cos model name")
    parser.add_argument("--patch-size", type=int, default=16, help="Square patch size")
    parser.add_argument("--generations", type=int, default=200, help="Number of ES generations")
    parser.add_argument("--population", type=int, default=50, help="Population size")
    parser.add_argument("--elite-fraction", type=float, default=0.2, help="Elite fraction")
    parser.add_argument("--sigma", type=float, default=0.08, help="Initial full-patch noise std")
    parser.add_argument("--sigma-decay", type=float, default=0.995, help="Sigma decay")
    parser.add_argument("--sigma-min", type=float, default=0.01, help="Minimum sigma")
    parser.add_argument("--lr", type=float, default=0.8, help="ES update rate")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.3,
        help="Perturbation budget. For l0, values <= 1 are treated as a fraction of patch coefficients.",
    )
    parser.add_argument("--norm", type=str, default="linf", help="Perturbation norm budget: l0, l2, or linf")
    parser.add_argument(
        "--score-batch-size",
        type=int,
        default=0,
        help="Total candidate evaluations per score forward. 0 = auto-chunk (safer for large image batches).",
    )
    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=128,
        help="Number of images to attack together in CSV mode. 0 = auto (8 on CUDA, 1 otherwise).",
    )
    parser.add_argument(
        "--attack-original-model",
        action="store_true",
        help="Use B-cos only to choose the patch position, then optimize/evaluate on the corresponding torchvision model.",
    )
    parser.add_argument(
        "--position-rule",
        choices=("margin", "top1", "dynamic-margin", "dynamic", "random"),
        default="margin",
        help=(
            "Patch position rule: margin = max patch sum(top1 contribution - top2 contribution), "
            "top1 = max patch sum(top1 contribution), "
            "dynamic-margin = max patch sum(top1 6-channel dynamic-weight norm - top2 6-channel dynamic-weight norm), "
            "dynamic = max patch sum(top1 6-channel dynamic-weight norm), "
            "random = random top-left patch position."
        ),
    )
    parser.add_argument("--save-images", action="store_true", help="Save before/after PNGs for each image")
    parser.add_argument("--save-figure", action="store_true", help="Render and save the explanation figure for each image")
    parser.add_argument("--verbose-generations", action="store_true", help="Print one log line per ES generation")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULT_DIR / "explain_guided_outputs"),
        help="Root folder for all generated outputs",
    )
    parser.add_argument("--save", type=str, default="explain_guided_es_patch.png", help="Output plot filename or path")
    args = parser.parse_args()

    if args.image and args.images_csv:
        parser.error("Use either --image or --images-csv, not both.")
    if not args.image and not args.images_csv:
        parser.error("Provide --image or --images-csv.")
    if args.image_batch_size < 0:
        parser.error("--image-batch-size must be >= 0.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    configure_fast_runtime(device)
    args.device_obj = device
    args.channels_last = device.type == "cuda"
    if args.image_batch_size == 0:
        args.image_batch_size = 8 if args.images_csv and device.type == "cuda" else 1

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Explain-guided ES Patch Attack")
    print("=" * 72)
    print(f"  Device       : {device}")
    print(f"  Model        : {args.model}")
    print(f"  Patch size   : {args.patch_size}x{args.patch_size}")
    print(f"  Budget       : {canonicalize_norm(args.norm)} <= {args.epsilon:g}")
    print(f"  Attack mode  : {'torchvision-original' if args.attack_original_model else 'bcos'}")
    position_rule_text = {
        "margin": "max patch sum(top1 contribution - top2 contribution)",
        "top1": "max patch sum(top1 contribution)",
        "dynamic-margin": "max patch sum(top1 6-channel dynamic-weight norm - top2 6-channel dynamic-weight norm)",
        "dynamic": "max patch sum(top1 6-channel dynamic-weight norm)",
        "random": "random top-left patch position",
    }[args.position_rule]
    print(f"  Position rule: {args.position_rule} ({position_rule_text})")
    if args.attack_original_model:
        print("  Guide only   : B-cos explain chooses position; ES optimize/eval run on torchvision model")
    print(f"  Image batch  : {args.image_batch_size}")
    print(f"  Score batch  : {'auto' if args.score_batch_size == 0 else args.score_batch_size}")
    print(f"  Verbose gen  : {args.verbose_generations}")
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
        attack_model, attack_model_name, ORIGINAL_ATTACK_PREPROCESS = load_original_attack_model(
            args.model,
            device=device,
            channels_last=args.channels_last,
        )
        print(f"  Using torchvision model: {attack_model_name}")
        print(f"  Original input: Resize({IMAGENET_SL}) -> CenterCrop({IMAGENET_SL}) -> ToTensor()")
        print("  Normalize     : applied after perturbation inside model_forward_original")
    else:
        ORIGINAL_ATTACK_PREPROCESS = None
        attack_model = guide_model

    if args.images_csv:
        image_items = load_image_paths_from_csv(Path(args.images_csv))
        print(f"\n▸ Loaded {len(image_items)} image paths from {args.images_csv}")
    else:
        image_items = [(None, args.image)]

    results: List[Dict[str, Any]] = []
    total = len(image_items)
    if args.images_csv and args.image_batch_size > 1:
        for start in range(0, total, args.image_batch_size):
            end = min(start + args.image_batch_size, total)
            batch_items = image_items[start:end]
            print("\n" + "-" * 72)
            print(f"  Images {start + 1}-{end}/{total}")
            print("-" * 72)
            results.extend(process_image_batch(guide_model, attack_model, batch_items, output_root, args))
    else:
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
            "position_rule",
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
