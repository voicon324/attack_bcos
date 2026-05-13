"""
Explain-guided ES patch attack with circle RGB perturbation parameters.

This version reuses the position-selection path from
explain_guided_pixel_es_patch.py, but ES no longer evolves every patch pixel.
Each candidate evolves N circles, where each circle is:

    (x, y, r, alpha, red, green, blue)

with x/y in patch coordinates, r as radius, alpha as opacity, and RGB as
an additive perturbation. Rendering a candidate starts from the original
image crop, applies the circle perturbations sequentially, then pastes the
final crop back into the image.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F
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


@dataclass(frozen=True)
class CircleCandidateEvaluation:
    score: float
    target_logit: float
    pred_class: int
    adversarial: bool
    l2: float
    pos_y: int
    pos_x: int


@dataclass(frozen=True)
class BatchedCircleCandidateEvaluation:
    scores: Tensor
    target_logits: Tensor
    pred_classes: Tensor
    adversarial: Tensor
    l2: Tensor
    pos_y: List[int]
    pos_x: List[int]


ORIGINAL_ATTACK_PREPROCESS: Optional[OriginalAttackPreprocess] = None
_BCOS_MODULE: Any = None
_EXTRACT_ATTRIBUTION: Any = None
_LOAD_RGB_IMAGE: Any = None
_TO_BCOS_INPUT: Any = None
_VISUALIZE_PATCH_RESULTS: Any = None


def ensure_bcos_runtime() -> None:
    global _BCOS_MODULE
    global _EXTRACT_ATTRIBUTION
    global _LOAD_RGB_IMAGE
    global _TO_BCOS_INPUT
    global _VISUALIZE_PATCH_RESULTS

    if _BCOS_MODULE is not None:
        return

    try:
        import bcos as bcos_module
        from bcos_es_patch import (
            extract_attribution as es_extract_attribution,
            load_rgb_image as es_load_rgb_image,
            to_bcos_input as es_to_bcos_input,
            visualize_patch_results as es_visualize_patch_results,
        )
    except Exception as exc:
        try:
            import importlib.metadata as metadata

            torchvision_version = metadata.version("torchvision")
        except Exception:
            torchvision_version = "unknown"
        raise RuntimeError(
            "Failed to import the B-cos runtime. In this conda env, "
            f"torch={torch.__version__} and torchvision={torchvision_version}; "
            "these versions must be compatible before running the attack."
        ) from exc

    _BCOS_MODULE = bcos_module
    _EXTRACT_ATTRIBUTION = es_extract_attribution
    _LOAD_RGB_IMAGE = es_load_rgb_image
    _TO_BCOS_INPUT = es_to_bcos_input
    _VISUALIZE_PATCH_RESULTS = es_visualize_patch_results


def extract_attribution(*args: Any, **kwargs: Any) -> Any:
    ensure_bcos_runtime()
    return _EXTRACT_ATTRIBUTION(*args, **kwargs)


def load_rgb_image(*args: Any, **kwargs: Any) -> Any:
    ensure_bcos_runtime()
    return _LOAD_RGB_IMAGE(*args, **kwargs)


def to_bcos_input(*args: Any, **kwargs: Any) -> Any:
    ensure_bcos_runtime()
    return _TO_BCOS_INPUT(*args, **kwargs)


def visualize_patch_results(*args: Any, **kwargs: Any) -> Any:
    ensure_bcos_runtime()
    return _VISUALIZE_PATCH_RESULTS(*args, **kwargs)


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


def resolve_original_model_weights(model_name: str) -> Any:
    from torchvision import models as tv_models

    weights_enum = tv_models.get_model_weights(model_name)
    weight_v1 = getattr(weights_enum, "IMAGENET1K_V1", None)
    if weight_v1 is not None:
        return weight_v1
    print(
        f"  warning      : {model_name} does not provide IMAGENET1K_V1; "
        f"falling back to DEFAULT"
    )
    return weights_enum.DEFAULT


def iter_original_weight_candidates(filename: str) -> List[Path]:
    candidates: List[Path] = []

    env_file = os.environ.get("ORIGINAL_WEIGHT_FILE") or os.environ.get("TORCHVISION_WEIGHT_FILE")
    if env_file:
        candidates.append(Path(env_file))

    roots: List[Path] = []
    for env_name in ("ORIGINAL_WEIGHTS_DIR", "TORCHVISION_WEIGHTS_DIR", "MODEL_WEIGHTS_DIR", "WEIGHTS_DIR"):
        env_dir = os.environ.get(env_name)
        if env_dir:
            roots.append(Path(env_dir))

    roots.extend(
        [
            SCRIPT_DIR / "weights",
            SCRIPT_DIR / "weights" / "torchvision-imagenet",
            SCRIPT_DIR / "artifacts" / "model-weights" / "weights",
            SCRIPT_DIR / "artifacts" / "model-weights" / "weights" / "torchvision-imagenet",
            Path("/kaggle/working/weights"),
            Path("/kaggle/working/weights/torchvision-imagenet"),
            Path("/kaggle/working/torchvision-imagenet"),
            Path("/kaggle/input"),
        ]
    )

    for root in roots:
        candidates.extend(
            [
                root / filename,
                root / "torchvision-imagenet" / filename,
                root / "weights" / filename,
                root / "weights" / "torchvision-imagenet" / filename,
                root / "torchvision-imagenet.zip",
                root / "weights" / "torchvision-imagenet.zip",
            ]
        )

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for dataset_dir in sorted(p for p in kaggle_input.rglob("*") if p.is_dir()):
            if dataset_dir.name in {"weights", "weights-bcos", "torchvision-imagenet"}:
                candidates.extend(
                    [
                        dataset_dir / filename,
                        dataset_dir / "torchvision-imagenet" / filename,
                        dataset_dir / "weights" / "torchvision-imagenet" / filename,
                        dataset_dir / "torchvision-imagenet.zip",
                        dataset_dir / "weights" / "torchvision-imagenet.zip",
                    ]
                )

    seen = set()
    unique: List[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def torch_load_cpu_bytes(data: bytes) -> Any:
    try:
        return torch.load(BytesIO(data), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(BytesIO(data), map_location="cpu")


def load_state_dict_from_zip(zip_path: Path, filename: str) -> Optional[Dict[str, Tensor]]:
    with zipfile.ZipFile(zip_path) as archive:
        preferred = [
            filename,
            f"torchvision-imagenet/{filename}",
            f"weights/torchvision-imagenet/{filename}",
        ]
        names = set(archive.namelist())
        for name in preferred:
            if name in names:
                return torch_load_cpu_bytes(archive.read(name))

        matches = [name for name in archive.namelist() if Path(name).name == filename]
        if matches:
            return torch_load_cpu_bytes(archive.read(matches[0]))
    return None


def find_original_model_state_dict(weights: Any) -> Optional[Tuple[Dict[str, Tensor], Path]]:
    url = getattr(weights, "url", "")
    if not url:
        return None
    filename = Path(urlparse(url).path).name
    if not filename:
        return None

    for candidate in iter_original_weight_candidates(filename):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() == ".zip":
            state_dict = load_state_dict_from_zip(candidate, filename)
            if state_dict is None:
                continue
            return state_dict, candidate
        if candidate.name == filename:
            return torch_load_cpu(candidate), candidate
    return None


def build_original_attack_preprocess(weights: Any) -> OriginalAttackPreprocess:
    from torchvision import transforms

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
    from torchvision import models as tv_models

    torchvision_name = resolve_torchvision_model_name(model_name)
    weights = resolve_original_model_weights(torchvision_name)
    preprocess = build_original_attack_preprocess(weights)
    local_state = find_original_model_state_dict(weights)
    if local_state is None:
        model = tv_models.get_model(torchvision_name, weights=weights)
    else:
        state_dict, weight_path = local_state
        print(f"  Loading local torchvision weights from {weight_path}")
        model = tv_models.get_model(torchvision_name, weights=None)
        model.load_state_dict(state_dict)
    model = model.to(device).eval()
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

    from PIL import Image

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


def find_best_patch_position(
    contribution_map: Tensor,
    patch_size: int,
) -> Tuple[int, int]:
    cmap = squeeze_contribution_map(contribution_map)
    return find_best_patch_position_from_importance_map(cmap, patch_size)


def resolve_patch_positions(
    primary_contribution_map: Tensor,
    secondary_contribution_map: Tensor,
    patch_size: int,
    pos_y: int,
    pos_x: int,
    use_explain_position: bool,
) -> List[Tuple[int, int]]:
    if use_explain_position:
        primary_pos = find_best_patch_position_from_contribution_margin(
            primary_contribution_map,
            secondary_contribution_map,
            patch_size,
        )
    else:
        primary_pos = (max(0, pos_y), max(0, pos_x))
    return [primary_pos]


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


class GuidedHelpers:
    RESULT_DIR = RESULT_DIR
    IMAGENET_SL = IMAGENET_SL
    configure_fast_runtime = staticmethod(configure_fast_runtime)
    maybe_channels_last = staticmethod(maybe_channels_last)
    model_forward_attack = staticmethod(model_forward_attack)
    load_original_attack_model = staticmethod(load_original_attack_model)
    load_attack_rgb_image = staticmethod(load_attack_rgb_image)
    compute_cw_scores = staticmethod(compute_cw_scores)
    resolve_runner_up_classes = staticmethod(resolve_runner_up_classes)
    resolve_total_candidate_batch_size = staticmethod(resolve_total_candidate_batch_size)
    extract_attribution = staticmethod(extract_attribution)
    to_bcos_input = staticmethod(to_bcos_input)
    visualize_patch_results = staticmethod(visualize_patch_results)
    resolve_patch_positions = staticmethod(resolve_patch_positions)
    load_rgb_image_batch = staticmethod(load_rgb_image_batch)
    load_image_paths_from_csv = staticmethod(load_image_paths_from_csv)

    @property
    def bcos(self) -> Any:
        ensure_bcos_runtime()
        return _BCOS_MODULE

    @property
    def ORIGINAL_ATTACK_PREPROCESS(self) -> Optional[OriginalAttackPreprocess]:
        return ORIGINAL_ATTACK_PREPROCESS

    @ORIGINAL_ATTACK_PREPROCESS.setter
    def ORIGINAL_ATTACK_PREPROCESS(self, value: Optional[OriginalAttackPreprocess]) -> None:
        global ORIGINAL_ATTACK_PREPROCESS
        ORIGINAL_ATTACK_PREPROCESS = value


guided = GuidedHelpers()


def make_patch_grid(patch_size: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
    coords = torch.arange(patch_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return yy.view(1, 1, patch_size, patch_size), xx.view(1, 1, patch_size, patch_size)


def decode_circle_params(
    normalized_circles: Tensor,
    patch_size: int,
    min_radius: float,
    max_radius: float,
    min_alpha: float,
    max_alpha: float,
    rgb_epsilon: float,
) -> Tensor:
    circles = normalized_circles.clamp(0.0, 1.0)
    decoded = torch.empty_like(circles)
    decoded[..., 0] = circles[..., 0] * max(patch_size - 1, 1)
    decoded[..., 1] = circles[..., 1] * max(patch_size - 1, 1)
    decoded[..., 2] = min_radius + circles[..., 2] * (max_radius - min_radius)
    decoded[..., 3] = min_alpha + circles[..., 3] * (max_alpha - min_alpha)
    decoded[..., 4:7] = (circles[..., 4:7] * 2.0 - 1.0) * rgb_epsilon
    return decoded


def circle_mask(
    grid_y: Tensor,
    grid_x: Tensor,
    center_x: Tensor,
    center_y: Tensor,
    radius: Tensor,
    edge_softness: float,
) -> Tensor:
    dist = ((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2).sqrt()
    if edge_softness <= 0:
        return (dist <= radius).to(dtype=grid_x.dtype)
    return torch.clamp((radius - dist) / max(edge_softness, 1e-6) + 0.5, 0.0, 1.0)


def render_circle_perturb_patch(
    x_rgb: Tensor,
    decoded_circles: Tensor,
    pos_y: int,
    pos_x: int,
    patch_size: int,
    grid_y: Tensor,
    grid_x: Tensor,
    edge_softness: float,
    linf_epsilon: float,
) -> Tensor:
    original_crop = x_rgb[:, :, pos_y:pos_y + patch_size, pos_x:pos_x + patch_size]
    crop = original_crop.clone()
    for circle_idx in range(decoded_circles.shape[0]):
        circle = decoded_circles[circle_idx]
        mask = circle_mask(
            grid_y,
            grid_x,
            center_x=circle[0],
            center_y=circle[1],
            radius=circle[2],
            edge_softness=edge_softness,
        )
        alpha = circle[3].view(1, 1, 1, 1)
        rgb_delta = circle[4:7].view(1, 3, 1, 1)
        crop = crop + mask * alpha * rgb_delta
    if linf_epsilon >= 0:
        crop = torch.max(torch.min(crop, original_crop + linf_epsilon), original_crop - linf_epsilon)
    return crop.clamp(0.0, 1.0)


def paste_patch(x_rgb: Tensor, patch_rgb: Tensor, pos_y: int, pos_x: int) -> Tensor:
    x_pert = x_rgb.clone()
    patch_size = patch_rgb.shape[-1]
    x_pert[:, :, pos_y:pos_y + patch_size, pos_x:pos_x + patch_size] = patch_rgb
    return x_pert


def paste_patches_batch(
    x_rgb: Tensor,
    patches_rgb: Tensor,
    pos_y: List[int],
    pos_x: List[int],
) -> Tensor:
    x_pert = x_rgb.clone()
    patch_size = patches_rgb.shape[-1]
    for image_idx in range(x_rgb.shape[0]):
        y = int(pos_y[image_idx])
        x = int(pos_x[image_idx])
        x_pert[image_idx:image_idx + 1, :, y:y + patch_size, x:x + patch_size] = patches_rgb[
            image_idx:image_idx + 1
        ]
    return x_pert


def render_patches_batch(
    x_rgb: Tensor,
    circles: Tensor,
    pos_y: List[int],
    pos_x: List[int],
    patch_size: int,
    min_radius: float,
    max_radius: float,
    min_alpha: float,
    max_alpha: float,
    rgb_epsilon: float,
    linf_epsilon: float,
    edge_softness: float,
) -> Tensor:
    grid_y, grid_x = make_patch_grid(patch_size, x_rgb.device, x_rgb.dtype)
    decoded = decode_circle_params(
        circles,
        patch_size=patch_size,
        min_radius=min_radius,
        max_radius=max_radius,
        min_alpha=min_alpha,
        max_alpha=max_alpha,
        rgb_epsilon=rgb_epsilon,
    )
    patches: List[Tensor] = []
    for image_idx in range(x_rgb.shape[0]):
        patch = render_circle_perturb_patch(
            x_rgb[image_idx:image_idx + 1],
            decoded[image_idx],
            int(pos_y[image_idx]),
            int(pos_x[image_idx]),
            patch_size,
            grid_y,
            grid_x,
            edge_softness=edge_softness,
            linf_epsilon=linf_epsilon,
        )
        patches.append(patch)
    return torch.cat(patches, dim=0)


def patch_rgb_for_visualization(patch_rgb: Tensor) -> Tensor:
    patch_4d = patch_rgb
    while patch_4d.dim() > 4 and patch_4d.shape[0] == 1:
        patch_4d = patch_4d.squeeze(0)
    if patch_4d.dim() == 3:
        patch_4d = patch_4d.unsqueeze(0)
    if patch_4d.dim() != 4 or patch_4d.shape[1] != 3:
        raise ValueError(f"Expected RGB patch, got shape {tuple(patch_rgb.shape)}")
    return patch_4d.contiguous() * 2.0 - 1.0


def circles_to_json(
    circles: Tensor,
    patch_size: int,
    min_radius: float,
    max_radius: float,
    min_alpha: float,
    max_alpha: float,
    rgb_epsilon: float,
) -> str:
    decoded = decode_circle_params(
        circles,
        patch_size=patch_size,
        min_radius=min_radius,
        max_radius=max_radius,
        min_alpha=min_alpha,
        max_alpha=max_alpha,
        rgb_epsilon=rgb_epsilon,
    ).detach().cpu()
    items = [
        {
            "x": round(float(circle[0].item()), 4),
            "y": round(float(circle[1].item()), 4),
            "r": round(float(circle[2].item()), 4),
            "alpha": round(float(circle[3].item()), 4),
            "red": round(float(circle[4].item()), 6),
            "green": round(float(circle[5].item()), 6),
            "blue": round(float(circle[6].item()), 6),
        }
        for circle in decoded
    ]
    return json.dumps(items, separators=(",", ":"))


def patch_l2_distance(x_rgb: Tensor, patch_rgb: Tensor, pos_y: int, pos_x: int) -> Tensor:
    patch_size = patch_rgb.shape[-1]
    original_crop = x_rgb[:, :, pos_y:pos_y + patch_size, pos_x:pos_x + patch_size]
    return ((patch_rgb - original_crop) ** 2).sum(dim=(1, 2, 3))


def evaluate_circles_multi_position(
    model: torch.nn.Module,
    x_rgb: Tensor,
    circles: Tensor,
    patch_positions: List[Tuple[int, int]],
    target_class: int,
    patch_size: int,
    min_radius: float,
    max_radius: float,
    min_alpha: float,
    max_alpha: float,
    rgb_epsilon: float,
    linf_epsilon: float,
    edge_softness: float,
    channels_last: bool,
    attack_original_model: bool,
) -> CircleCandidateEvaluation:
    if circles.dim() != 2:
        raise ValueError(f"Expected one circle genotype with shape (N, 7), got {tuple(circles.shape)}")

    grid_y, grid_x = make_patch_grid(patch_size, x_rgb.device, x_rgb.dtype)
    decoded = decode_circle_params(
        circles,
        patch_size=patch_size,
        min_radius=min_radius,
        max_radius=max_radius,
        min_alpha=min_alpha,
        max_alpha=max_alpha,
        rgb_epsilon=rgb_epsilon,
    )

    position_count = len(patch_positions)
    x_rep = x_rgb.expand(position_count, -1, -1, -1).clone()
    l2_values: List[Tensor] = []
    for pos_idx, (pos_y, pos_x) in enumerate(patch_positions):
        patch = render_circle_perturb_patch(
            x_rgb,
            decoded,
            pos_y,
            pos_x,
            patch_size,
            grid_y,
            grid_x,
            edge_softness=edge_softness,
            linf_epsilon=linf_epsilon,
        )
        x_rep[pos_idx:pos_idx + 1, :, pos_y:pos_y + patch_size, pos_x:pos_x + patch_size] = patch
        l2_values.append(patch_l2_distance(x_rgb, patch, pos_y, pos_x).squeeze(0))

    x_rep = guided.maybe_channels_last(x_rep, channels_last)
    with torch.inference_mode():
        outputs = guided.model_forward_attack(
            model,
            x_rep,
            channels_last=channels_last,
            attack_original_model=attack_original_model,
        )
        targets = torch.full(
            (outputs.shape[0],),
            target_class,
            device=outputs.device,
            dtype=torch.long,
        )
        scores = guided.compute_cw_scores(outputs, targets)

    best_pos_idx = int(scores.argmax().item())
    best_output = outputs[best_pos_idx]
    best_pos = patch_positions[best_pos_idx]
    pred_class = int(best_output.argmax().item())
    return CircleCandidateEvaluation(
        score=float(scores[best_pos_idx].item()),
        target_logit=float(best_output[target_class].item()),
        pred_class=pred_class,
        adversarial=pred_class != target_class,
        l2=float(torch.stack(l2_values)[best_pos_idx].item()),
        pos_y=best_pos[0],
        pos_x=best_pos[1],
    )


def evaluate_circles_batched_multi_position(
    model: torch.nn.Module,
    x_rgb: Tensor,
    circles: Tensor,
    patch_positions: List[List[Tuple[int, int]]],
    target_classes: Tensor,
    patch_size: int,
    min_radius: float,
    max_radius: float,
    min_alpha: float,
    max_alpha: float,
    rgb_epsilon: float,
    linf_epsilon: float,
    edge_softness: float,
    channels_last: bool,
    attack_original_model: bool,
) -> BatchedCircleCandidateEvaluation:
    if circles.dim() != 3:
        raise ValueError(f"Expected batched circle genotypes with shape (B, N, 7), got {tuple(circles.shape)}")

    batch_size = x_rgb.shape[0]
    position_count = len(patch_positions[0]) if patch_positions else 1
    grid_y, grid_x = make_patch_grid(patch_size, x_rgb.device, x_rgb.dtype)
    decoded = decode_circle_params(
        circles,
        patch_size=patch_size,
        min_radius=min_radius,
        max_radius=max_radius,
        min_alpha=min_alpha,
        max_alpha=max_alpha,
        rgb_epsilon=rgb_epsilon,
    )

    x_rep = (
        x_rgb.unsqueeze(1)
        .expand(-1, position_count, -1, -1, -1)
        .clone()
        .reshape(batch_size * position_count, *x_rgb.shape[1:])
    )
    l2_values = torch.empty(batch_size, position_count, device=x_rgb.device, dtype=torch.float32)

    for image_idx in range(batch_size):
        source = x_rgb[image_idx:image_idx + 1]
        for pos_idx, (pos_y, pos_x) in enumerate(patch_positions[image_idx]):
            patch = render_circle_perturb_patch(
                source,
                decoded[image_idx],
                pos_y,
                pos_x,
                patch_size,
                grid_y,
                grid_x,
                edge_softness=edge_softness,
                linf_epsilon=linf_epsilon,
            )
            row_idx = image_idx * position_count + pos_idx
            x_rep[row_idx:row_idx + 1, :, pos_y:pos_y + patch_size, pos_x:pos_x + patch_size] = patch
            l2_values[image_idx, pos_idx] = patch_l2_distance(source, patch, pos_y, pos_x).squeeze(0).float()

    x_rep = guided.maybe_channels_last(x_rep, channels_last)
    with torch.inference_mode():
        outputs = guided.model_forward_attack(
            model,
            x_rep,
            channels_last=channels_last,
            attack_original_model=attack_original_model,
        ).view(batch_size, position_count, -1)
        flat_outputs = outputs.reshape(batch_size * position_count, -1)
        flat_targets = target_classes.unsqueeze(1).expand(-1, position_count).reshape(-1)
        flat_scores = guided.compute_cw_scores(flat_outputs, flat_targets)
        scores = flat_scores.view(batch_size, position_count)

    best_scores, best_pos_idx = scores.max(dim=1)
    image_idx = torch.arange(batch_size, device=outputs.device)
    best_outputs = outputs[image_idx, best_pos_idx]
    target_logits = best_outputs.gather(1, target_classes.unsqueeze(1)).squeeze(1)
    pred_classes = best_outputs.argmax(dim=1)
    best_l2 = l2_values.gather(1, best_pos_idx.unsqueeze(1)).squeeze(1)
    best_pos_y = [
        patch_positions[idx][int(best_pos_idx[idx].item())][0]
        for idx in range(batch_size)
    ]
    best_pos_x = [
        patch_positions[idx][int(best_pos_idx[idx].item())][1]
        for idx in range(batch_size)
    ]

    return BatchedCircleCandidateEvaluation(
        scores=best_scores.float(),
        target_logits=target_logits.float(),
        pred_classes=pred_classes,
        adversarial=pred_classes != target_classes,
        l2=best_l2.float(),
        pos_y=best_pos_y,
        pos_x=best_pos_x,
    )


def camo_mutate_circles(circles: Tensor, mutation_rate: float) -> Tensor:
    mutated = circles.clone()
    genes, length = mutated.shape
    if genes == 0 or length == 0:
        return mutated

    gene_idx = int(torch.randint(0, genes, (1,), device=mutated.device).item())
    change_count = int(torch.randint(0, length + 1, (1,), device=mutated.device).item())
    if change_count == 0:
        return mutated

    selection = torch.randperm(length, device=mutated.device)[:change_count]
    if float(torch.rand((), device=mutated.device).item()) < mutation_rate:
        mutated[gene_idx, selection] = torch.rand(
            change_count,
            device=mutated.device,
            dtype=mutated.dtype,
        )
    else:
        delta = (
            torch.rand(change_count, device=mutated.device, dtype=mutated.dtype) - 0.5
        ) / 3.0
        mutated[gene_idx, selection] = (mutated[gene_idx, selection] + delta).clamp(0.0, 1.0)
    return mutated.clamp(0.0, 1.0)


def camo_mutate_circles_batch(circles: Tensor, mutation_rate: float) -> Tensor:
    return torch.stack(
        [camo_mutate_circles(circles[idx], mutation_rate) for idx in range(circles.shape[0])],
        dim=0,
    )


def camo_accept_candidate(
    current: CircleCandidateEvaluation,
    candidate: CircleCandidateEvaluation,
) -> bool:
    if current.adversarial and candidate.adversarial:
        return candidate.l2 < current.l2
    return candidate.score > current.score


def camo_accept_mask(
    current_adversarial: Tensor,
    current_scores: Tensor,
    current_l2: Tensor,
    candidate_adversarial: Tensor,
    candidate_scores: Tensor,
    candidate_l2: Tensor,
) -> Tensor:
    both_adversarial = current_adversarial & candidate_adversarial
    return torch.where(
        both_adversarial,
        candidate_l2 < current_l2,
        candidate_scores > current_scores,
    )


class CirclePerturbPatchES:
    def __init__(
        self,
        num_circles: int,
        population_size: int,
        elite_fraction: float,
        sigma: float,
        sigma_decay: float,
        sigma_min: float,
        learning_rate: float,
        device: torch.device,
    ):
        self.num_circles = num_circles
        self.population_size = population_size
        self.elite_size = max(1, int(population_size * elite_fraction))
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min
        self.lr = learning_rate
        self.device = device
        self.params = torch.rand(num_circles, 7, device=device)

    def ask(self) -> Tensor:
        noise = torch.randn(self.population_size, self.num_circles, 7, device=self.device)
        return (self.params.unsqueeze(0) + self.sigma * noise).clamp(0.0, 1.0)

    def tell(self, candidates: Tensor, scores: Tensor) -> None:
        elite_idx = scores.topk(self.elite_size).indices
        elite = candidates[elite_idx]
        elite_scores = scores[elite_idx]
        elite_scores = (elite_scores - elite_scores.mean()) / (
            elite_scores.std(unbiased=False) + 1e-8
        )
        weights = F.softmax(elite_scores, dim=0)
        new_params = (weights.view(self.elite_size, 1, 1) * elite).sum(dim=0)
        self.params = ((1.0 - self.lr) * self.params + self.lr * new_params).clamp(0.0, 1.0)
        self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)

    def get_circles(self) -> Tensor:
        return self.params.clone()


class BatchedCirclePerturbPatchES:
    def __init__(
        self,
        batch_size: int,
        num_circles: int,
        population_size: int,
        elite_fraction: float,
        sigma: float,
        sigma_decay: float,
        sigma_min: float,
        learning_rate: float,
        device: torch.device,
    ):
        self.batch_size = batch_size
        self.num_circles = num_circles
        self.population_size = population_size
        self.elite_size = max(1, int(population_size * elite_fraction))
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min
        self.lr = learning_rate
        self.device = device
        self.params = torch.rand(batch_size, num_circles, 7, device=device)

    def ask(self) -> Tensor:
        noise = torch.randn(
            self.batch_size,
            self.population_size,
            self.num_circles,
            7,
            device=self.device,
        )
        return (self.params.unsqueeze(1) + self.sigma * noise).clamp(0.0, 1.0)

    def tell(self, candidates: Tensor, scores: Tensor) -> None:
        elite_idx = scores.topk(self.elite_size, dim=1).indices
        gather_idx = elite_idx.view(self.batch_size, self.elite_size, 1, 1).expand(
            -1,
            -1,
            self.num_circles,
            7,
        )
        elite = candidates.gather(1, gather_idx)
        elite_scores = scores.gather(1, elite_idx)
        elite_scores = (
            elite_scores - elite_scores.mean(dim=1, keepdim=True)
        ) / (elite_scores.std(dim=1, keepdim=True, unbiased=False) + 1e-8)
        weights = F.softmax(elite_scores, dim=1)
        new_params = (weights.view(self.batch_size, self.elite_size, 1, 1) * elite).sum(dim=1)
        self.params = ((1.0 - self.lr) * self.params + self.lr * new_params).clamp(0.0, 1.0)
        self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)

    def get_circles(self) -> Tensor:
        return self.params.clone()


def compute_scores_fixed_multi_position(
    model: torch.nn.Module,
    x_rgb: Tensor,
    candidate_circles: Tensor,
    patch_positions: List[Tuple[int, int]],
    target_class: int,
    patch_size: int,
    min_radius: float,
    max_radius: float,
    min_alpha: float,
    max_alpha: float,
    rgb_epsilon: float,
    linf_epsilon: float,
    edge_softness: float,
    score_batch_size: int,
    channels_last: bool,
    attack_original_model: bool,
) -> Tuple[Tensor, Tensor]:
    n = candidate_circles.shape[0]
    position_count = len(patch_positions)
    total_eval_batch = guided.resolve_total_candidate_batch_size(
        x_rgb,
        score_batch_size=score_batch_size,
        attack_original_model=attack_original_model,
    )
    batch_size = min(n, max(1, total_eval_batch // max(position_count, 1)))
    scores_list: List[Tensor] = []
    pos_idx_list: List[Tensor] = []
    grid_y, grid_x = make_patch_grid(patch_size, x_rgb.device, x_rgb.dtype)

    with torch.inference_mode():
        for start in range(0, n, batch_size):
            circle_batch = candidate_circles[start:start + batch_size]
            decoded_batch = decode_circle_params(
                circle_batch,
                patch_size=patch_size,
                min_radius=min_radius,
                max_radius=max_radius,
                min_alpha=min_alpha,
                max_alpha=max_alpha,
                rgb_epsilon=rgb_epsilon,
            )
            batch = circle_batch.shape[0]

            x_rep = x_rgb.expand(batch * position_count, -1, -1, -1).clone()
            for pos_idx, (pos_y, pos_x) in enumerate(patch_positions):
                for cand_idx in range(batch):
                    row_idx = pos_idx * batch + cand_idx
                    patch = render_circle_perturb_patch(
                        x_rgb,
                        decoded_batch[cand_idx],
                        pos_y,
                        pos_x,
                        patch_size,
                        grid_y,
                        grid_x,
                        edge_softness=edge_softness,
                        linf_epsilon=linf_epsilon,
                    )
                    x_rep[row_idx:row_idx + 1, :, pos_y:pos_y + patch_size, pos_x:pos_x + patch_size] = patch

            x_rep = guided.maybe_channels_last(x_rep, channels_last)
            outputs = guided.model_forward_attack(
                model,
                x_rep,
                channels_last=channels_last,
                attack_original_model=attack_original_model,
            )
            targets = torch.full(
                (outputs.shape[0],),
                target_class,
                device=outputs.device,
                dtype=torch.long,
            )
            per_position_scores = guided.compute_cw_scores(outputs, targets).view(position_count, batch)
            best_scores, best_pos_idx = per_position_scores.max(dim=0)
            scores_list.append(best_scores.float())
            pos_idx_list.append(best_pos_idx.to(dtype=torch.long))

    return torch.cat(scores_list), torch.cat(pos_idx_list)


def compute_scores_fixed_batched_multi_position(
    model: torch.nn.Module,
    x_rgb: Tensor,
    candidate_circles: Tensor,
    patch_positions: List[List[Tuple[int, int]]],
    target_classes: Tensor,
    patch_size: int,
    min_radius: float,
    max_radius: float,
    min_alpha: float,
    max_alpha: float,
    rgb_epsilon: float,
    linf_epsilon: float,
    edge_softness: float,
    score_batch_size: int,
    channels_last: bool,
    attack_original_model: bool,
) -> Tuple[Tensor, Tensor]:
    batch_size, population_size = candidate_circles.shape[:2]
    position_count = len(patch_positions[0]) if patch_positions else 1
    total_eval_batch = guided.resolve_total_candidate_batch_size(
        x_rgb,
        score_batch_size=score_batch_size,
        attack_original_model=attack_original_model,
    )
    pop_chunk = max(
        1,
        min(population_size, total_eval_batch // max(batch_size * position_count, 1)),
    )
    scores_list: List[Tensor] = []
    pos_idx_list: List[Tensor] = []
    grid_y, grid_x = make_patch_grid(patch_size, x_rgb.device, x_rgb.dtype)

    with torch.inference_mode():
        for start in range(0, population_size, pop_chunk):
            circle_chunk = candidate_circles[:, start:start + pop_chunk]
            decoded_chunk = decode_circle_params(
                circle_chunk,
                patch_size=patch_size,
                min_radius=min_radius,
                max_radius=max_radius,
                min_alpha=min_alpha,
                max_alpha=max_alpha,
                rgb_epsilon=rgb_epsilon,
            )
            chunk_size = circle_chunk.shape[1]

            x_rep = (
                x_rgb.unsqueeze(1)
                .unsqueeze(2)
                .expand(-1, position_count, chunk_size, -1, -1, -1)
                .clone()
                .reshape(batch_size * position_count * chunk_size, *x_rgb.shape[1:])
            )
            for image_idx in range(batch_size):
                source = x_rgb[image_idx:image_idx + 1]
                for pos_idx, (pos_y, pos_x) in enumerate(patch_positions[image_idx]):
                    for cand_idx in range(chunk_size):
                        row_idx = (image_idx * position_count + pos_idx) * chunk_size + cand_idx
                        patch = render_circle_perturb_patch(
                            source,
                            decoded_chunk[image_idx, cand_idx],
                            pos_y,
                            pos_x,
                            patch_size,
                            grid_y,
                            grid_x,
                            edge_softness=edge_softness,
                            linf_epsilon=linf_epsilon,
                        )
                        x_rep[row_idx:row_idx + 1, :, pos_y:pos_y + patch_size, pos_x:pos_x + patch_size] = patch

            x_rep = guided.maybe_channels_last(x_rep, channels_last)
            outputs = guided.model_forward_attack(
                model,
                x_rep,
                channels_last=channels_last,
                attack_original_model=attack_original_model,
            ).view(batch_size, position_count, chunk_size, -1)
            flat_outputs = outputs.reshape(batch_size * position_count * chunk_size, -1)
            flat_targets = target_classes.unsqueeze(1).unsqueeze(2).expand(
                -1,
                position_count,
                chunk_size,
            ).reshape(-1)
            flat_scores = guided.compute_cw_scores(flat_outputs, flat_targets)
            per_position_scores = flat_scores.view(batch_size, position_count, chunk_size)
            best_scores, best_pos_idx = per_position_scores.max(dim=1)
            scores_list.append(best_scores.float())
            pos_idx_list.append(best_pos_idx.to(dtype=torch.long))

    return torch.cat(scores_list, dim=1), torch.cat(pos_idx_list, dim=1)


def select_best_position_for_circles(
    model: torch.nn.Module,
    x_rgb: Tensor,
    circles: Tensor,
    patch_positions: List[Tuple[int, int]],
    target_class: int,
    patch_size: int,
    min_radius: float,
    max_radius: float,
    min_alpha: float,
    max_alpha: float,
    rgb_epsilon: float,
    linf_epsilon: float,
    edge_softness: float,
    score_batch_size: int,
    channels_last: bool,
    attack_original_model: bool,
) -> Tuple[int, int]:
    _, best_pos_idx = compute_scores_fixed_multi_position(
        model,
        x_rgb,
        circles.unsqueeze(0),
        patch_positions,
        target_class,
        patch_size=patch_size,
        min_radius=min_radius,
        max_radius=max_radius,
        min_alpha=min_alpha,
        max_alpha=max_alpha,
        rgb_epsilon=rgb_epsilon,
        linf_epsilon=linf_epsilon,
        edge_softness=edge_softness,
        score_batch_size=score_batch_size,
        channels_last=channels_last,
        attack_original_model=attack_original_model,
    )
    best_pos = patch_positions[int(best_pos_idx[0].item())]
    return best_pos[0], best_pos[1]


def run_explain_guided_camo_one_plus_one_attack(
    guide_model: torch.nn.Module,
    attack_model: torch.nn.Module,
    x_rgb: Tensor,
    target_class: int,
    secondary_class: int,
    patch_size: int,
    num_circles: int,
    generations: int,
    camo_mutation_rate: float,
    min_radius: float,
    max_radius: float,
    min_alpha: float,
    max_alpha: float,
    rgb_epsilon: float,
    linf_epsilon: float,
    edge_softness: float,
    pos_y: int,
    pos_x: int,
    use_explain_position: bool,
    channels_last: bool,
    attack_original_model: bool,
    verbose: bool,
) -> Tuple[Tensor, Tensor, int, int, List[Tuple[int, float, float, float]]]:
    primary_guidance = guided.extract_attribution(
        guide_model,
        guided.to_bcos_input(x_rgb),
        target_class=target_class,
    )
    secondary_guidance = guided.extract_attribution(
        guide_model,
        guided.to_bcos_input(x_rgb),
        target_class=secondary_class,
    )
    patch_positions = guided.resolve_patch_positions(
        primary_guidance["contribution_map"],
        secondary_guidance["contribution_map"],
        patch_size,
        pos_y=pos_y,
        pos_x=pos_x,
        use_explain_position=use_explain_position,
    )

    current_circles = torch.rand(num_circles, 7, device=x_rgb.device)
    current_eval = evaluate_circles_multi_position(
        attack_model,
        x_rgb,
        current_circles,
        patch_positions,
        target_class,
        patch_size=patch_size,
        min_radius=min_radius,
        max_radius=max_radius,
        min_alpha=min_alpha,
        max_alpha=max_alpha,
        rgb_epsilon=rgb_epsilon,
        linf_epsilon=linf_epsilon,
        edge_softness=edge_softness,
        channels_last=channels_last,
        attack_original_model=attack_original_model,
    )

    total_queries = max(generations, 1)
    history: List[Tuple[int, float, float, float]] = [
        (0, current_eval.score, current_eval.target_logit, current_eval.l2)
    ]
    if verbose:
        print(
            f"  Query {1:4d}/{total_queries} | "
            f"Target Logit: {current_eval.target_logit:7.3f} | "
            f"Score: {current_eval.score:7.3f} | L2: {current_eval.l2:9.5f} | init"
        )

    for query in range(1, total_queries):
        candidate_circles = camo_mutate_circles(current_circles, camo_mutation_rate)
        candidate_eval = evaluate_circles_multi_position(
            attack_model,
            x_rgb,
            candidate_circles,
            patch_positions,
            target_class,
            patch_size=patch_size,
            min_radius=min_radius,
            max_radius=max_radius,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
            rgb_epsilon=rgb_epsilon,
            linf_epsilon=linf_epsilon,
            edge_softness=edge_softness,
            channels_last=channels_last,
            attack_original_model=attack_original_model,
        )

        accepted = camo_accept_candidate(current_eval, candidate_eval)
        if accepted:
            current_circles = candidate_circles
            current_eval = candidate_eval

        history.append((query, current_eval.score, current_eval.target_logit, current_eval.l2))
        if verbose:
            status = "accepted" if accepted else "kept"
            print(
                f"  Query {query + 1:4d}/{total_queries} | "
                f"Target Logit: {current_eval.target_logit:7.3f} | "
                f"Score: {current_eval.score:7.3f} | L2: {current_eval.l2:9.5f} | {status}"
            )

    grid_y, grid_x = make_patch_grid(patch_size, x_rgb.device, x_rgb.dtype)
    final_patch = render_circle_perturb_patch(
        x_rgb,
        decode_circle_params(
            current_circles,
            patch_size=patch_size,
            min_radius=min_radius,
            max_radius=max_radius,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
            rgb_epsilon=rgb_epsilon,
        ),
        current_eval.pos_y,
        current_eval.pos_x,
        patch_size,
        grid_y,
        grid_x,
        edge_softness=edge_softness,
        linf_epsilon=linf_epsilon,
    )
    return final_patch, current_circles, current_eval.pos_y, current_eval.pos_x, history


def run_explain_guided_camo_one_plus_one_attack_batch(
    guide_model: torch.nn.Module,
    attack_model: torch.nn.Module,
    x_rgb: Tensor,
    target_classes: Tensor,
    secondary_classes: Tensor,
    patch_size: int,
    num_circles: int,
    generations: int,
    camo_mutation_rate: float,
    min_radius: float,
    max_radius: float,
    min_alpha: float,
    max_alpha: float,
    rgb_epsilon: float,
    linf_epsilon: float,
    edge_softness: float,
    pos_y: int,
    pos_x: int,
    use_explain_position: bool,
    channels_last: bool,
    attack_original_model: bool,
    verbose: bool,
) -> Tuple[Tensor, Tensor, List[int], List[int], List[List[Tuple[int, float, float, float]]]]:
    batch_size = x_rgb.shape[0]
    patch_positions: List[List[Tuple[int, int]]] = []
    for image_idx in range(batch_size):
        target_class = int(target_classes[image_idx].item())
        secondary_class = int(secondary_classes[image_idx].item())
        primary_guidance = guided.extract_attribution(
            guide_model,
            guided.to_bcos_input(x_rgb[image_idx:image_idx + 1]),
            target_class=target_class,
        )
        secondary_guidance = guided.extract_attribution(
            guide_model,
            guided.to_bcos_input(x_rgb[image_idx:image_idx + 1]),
            target_class=secondary_class,
        )
        patch_positions.append(
            guided.resolve_patch_positions(
                primary_guidance["contribution_map"],
                secondary_guidance["contribution_map"],
                patch_size,
                pos_y=pos_y,
                pos_x=pos_x,
                use_explain_position=use_explain_position,
            )
        )

    current_circles = torch.rand(batch_size, num_circles, 7, device=x_rgb.device)
    current_eval = evaluate_circles_batched_multi_position(
        attack_model,
        x_rgb,
        current_circles,
        patch_positions,
        target_classes,
        patch_size=patch_size,
        min_radius=min_radius,
        max_radius=max_radius,
        min_alpha=min_alpha,
        max_alpha=max_alpha,
        rgb_epsilon=rgb_epsilon,
        linf_epsilon=linf_epsilon,
        edge_softness=edge_softness,
        channels_last=channels_last,
        attack_original_model=attack_original_model,
    )

    total_queries = max(generations, 1)
    histories: List[List[Tuple[int, float, float, float]]] = [[] for _ in range(batch_size)]
    for image_idx in range(batch_size):
        histories[image_idx].append(
            (
                0,
                float(current_eval.scores[image_idx].item()),
                float(current_eval.target_logits[image_idx].item()),
                float(current_eval.l2[image_idx].item()),
            )
        )

    if verbose:
        print(
            f"  Query {1:4d}/{total_queries} | "
            f"Mean Target Logit: {current_eval.target_logits.mean().item():7.3f} | "
            f"Mean Score: {current_eval.scores.mean().item():7.3f} | "
            f"Mean L2: {current_eval.l2.mean().item():9.5f} | init"
        )

    current_scores = current_eval.scores
    current_target_logits = current_eval.target_logits
    current_adversarial = current_eval.adversarial
    current_l2 = current_eval.l2
    current_pos_y = current_eval.pos_y
    current_pos_x = current_eval.pos_x

    for query in range(1, total_queries):
        candidate_circles = camo_mutate_circles_batch(current_circles, camo_mutation_rate)
        candidate_eval = evaluate_circles_batched_multi_position(
            attack_model,
            x_rgb,
            candidate_circles,
            patch_positions,
            target_classes,
            patch_size=patch_size,
            min_radius=min_radius,
            max_radius=max_radius,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
            rgb_epsilon=rgb_epsilon,
            linf_epsilon=linf_epsilon,
            edge_softness=edge_softness,
            channels_last=channels_last,
            attack_original_model=attack_original_model,
        )

        accepted = camo_accept_mask(
            current_adversarial,
            current_scores,
            current_l2,
            candidate_eval.adversarial,
            candidate_eval.scores,
            candidate_eval.l2,
        )
        update_mask = accepted.view(batch_size, 1, 1)
        current_circles = torch.where(update_mask, candidate_circles, current_circles)
        current_scores = torch.where(accepted, candidate_eval.scores, current_scores)
        current_target_logits = torch.where(
            accepted,
            candidate_eval.target_logits,
            current_target_logits,
        )
        current_adversarial = torch.where(
            accepted,
            candidate_eval.adversarial,
            current_adversarial,
        )
        current_l2 = torch.where(accepted, candidate_eval.l2, current_l2)
        for image_idx in range(batch_size):
            if bool(accepted[image_idx].item()):
                current_pos_y[image_idx] = candidate_eval.pos_y[image_idx]
                current_pos_x[image_idx] = candidate_eval.pos_x[image_idx]
            histories[image_idx].append(
                (
                    query,
                    float(current_scores[image_idx].item()),
                    float(current_target_logits[image_idx].item()),
                    float(current_l2[image_idx].item()),
                )
            )

        if verbose:
            print(
                f"  Query {query + 1:4d}/{total_queries} | "
                f"Mean Target Logit: {current_target_logits.mean().item():7.3f} | "
                f"Mean Score: {current_scores.mean().item():7.3f} | "
                f"Mean L2: {current_l2.mean().item():9.5f} | "
                f"accepted {int(accepted.sum().item())}/{batch_size}"
            )

    final_patches = render_patches_batch(
        x_rgb,
        current_circles,
        current_pos_y,
        current_pos_x,
        patch_size=patch_size,
        min_radius=min_radius,
        max_radius=max_radius,
        min_alpha=min_alpha,
        max_alpha=max_alpha,
        rgb_epsilon=rgb_epsilon,
        linf_epsilon=linf_epsilon,
        edge_softness=edge_softness,
    )
    return final_patches, current_circles, current_pos_y, current_pos_x, histories


def run_explain_guided_attack(
    guide_model: torch.nn.Module,
    attack_model: torch.nn.Module,
    x_rgb: Tensor,
    target_class: int,
    secondary_class: int,
    patch_size: int,
    num_circles: int,
    generations: int,
    population_size: int,
    elite_fraction: float,
    sigma: float,
    sigma_decay: float,
    sigma_min: float,
    learning_rate: float,
    min_radius: float,
    max_radius: float,
    min_alpha: float,
    max_alpha: float,
    rgb_epsilon: float,
    linf_epsilon: float,
    edge_softness: float,
    pos_y: int,
    pos_x: int,
    use_explain_position: bool,
    score_batch_size: int,
    channels_last: bool,
    attack_original_model: bool,
    verbose: bool,
) -> Tuple[Tensor, Tensor, int, int, List[Tuple[int, float, float, float]]]:
    primary_guidance = guided.extract_attribution(
        guide_model,
        guided.to_bcos_input(x_rgb),
        target_class=target_class,
    )
    secondary_guidance = guided.extract_attribution(
        guide_model,
        guided.to_bcos_input(x_rgb),
        target_class=secondary_class,
    )
    patch_positions = guided.resolve_patch_positions(
        primary_guidance["contribution_map"],
        secondary_guidance["contribution_map"],
        patch_size,
        pos_y=pos_y,
        pos_x=pos_x,
        use_explain_position=use_explain_position,
    )

    es = CirclePerturbPatchES(
        num_circles=num_circles,
        population_size=population_size,
        elite_fraction=elite_fraction,
        sigma=sigma,
        sigma_decay=sigma_decay,
        sigma_min=sigma_min,
        learning_rate=learning_rate,
        device=x_rgb.device,
    )

    history: List[Tuple[int, float, float, float]] = []
    best_circles_overall: Optional[Tensor] = None
    best_score_overall = -float("inf")
    best_pos_overall = patch_positions[0]
    grid_y, grid_x = make_patch_grid(patch_size, x_rgb.device, x_rgb.dtype)

    for gen in range(generations):
        candidate_circles = es.ask()
        scores, best_pos_idx = compute_scores_fixed_multi_position(
            attack_model,
            x_rgb,
            candidate_circles,
            patch_positions,
            target_class,
            patch_size=patch_size,
            min_radius=min_radius,
            max_radius=max_radius,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
            rgb_epsilon=rgb_epsilon,
            linf_epsilon=linf_epsilon,
            edge_softness=edge_softness,
            score_batch_size=score_batch_size,
            channels_last=channels_last,
            attack_original_model=attack_original_model,
        )
        es.tell(candidate_circles, scores)

        best_idx = scores.argmax().item()
        best_circles = candidate_circles[best_idx]
        best_pos = patch_positions[int(best_pos_idx[best_idx].item())]
        best_patch = render_circle_perturb_patch(
            x_rgb,
            decode_circle_params(
                best_circles,
                patch_size=patch_size,
                min_radius=min_radius,
                max_radius=max_radius,
                min_alpha=min_alpha,
                max_alpha=max_alpha,
                rgb_epsilon=rgb_epsilon,
            ),
            best_pos[0],
            best_pos[1],
            patch_size,
            grid_y,
            grid_x,
            edge_softness=edge_softness,
            linf_epsilon=linf_epsilon,
        )
        best_x = paste_patch(x_rgb, best_patch, best_pos[0], best_pos[1])
        with torch.inference_mode():
            out_best = guided.model_forward_attack(
                attack_model,
                best_x,
                channels_last=channels_last,
                attack_original_model=attack_original_model,
            )
            best_logit = out_best[0, target_class].item()

        best_score = scores[best_idx].item()
        if best_score > best_score_overall:
            best_score_overall = best_score
            best_circles_overall = best_circles.clone()
            best_pos_overall = best_pos
        history.append((gen, best_score, best_logit, es.sigma))
        if verbose:
            print(
                f"  Gen {gen + 1:3d}/{generations} | "
                f"Target Logit: {best_logit:7.3f} | "
                f"Score: {best_score:7.3f} | sigma: {es.sigma:.4f}"
            )

    final_circles = es.get_circles()
    if best_circles_overall is not None:
        final_circles = best_circles_overall.clone()
    else:
        best_pos_overall = select_best_position_for_circles(
            attack_model,
            x_rgb,
            final_circles,
            patch_positions,
            target_class,
            patch_size=patch_size,
            min_radius=min_radius,
            max_radius=max_radius,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
            rgb_epsilon=rgb_epsilon,
            linf_epsilon=linf_epsilon,
            edge_softness=edge_softness,
            score_batch_size=score_batch_size,
            channels_last=channels_last,
            attack_original_model=attack_original_model,
        )

    final_patch = render_circle_perturb_patch(
        x_rgb,
        decode_circle_params(
            final_circles,
            patch_size=patch_size,
            min_radius=min_radius,
            max_radius=max_radius,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
            rgb_epsilon=rgb_epsilon,
        ),
        best_pos_overall[0],
        best_pos_overall[1],
        patch_size,
        grid_y,
        grid_x,
        edge_softness=edge_softness,
        linf_epsilon=linf_epsilon,
    )
    return final_patch, final_circles, best_pos_overall[0], best_pos_overall[1], history


def run_explain_guided_attack_batch(
    guide_model: torch.nn.Module,
    attack_model: torch.nn.Module,
    x_rgb: Tensor,
    target_classes: Tensor,
    secondary_classes: Tensor,
    patch_size: int,
    num_circles: int,
    generations: int,
    population_size: int,
    elite_fraction: float,
    sigma: float,
    sigma_decay: float,
    sigma_min: float,
    learning_rate: float,
    min_radius: float,
    max_radius: float,
    min_alpha: float,
    max_alpha: float,
    rgb_epsilon: float,
    linf_epsilon: float,
    edge_softness: float,
    pos_y: int,
    pos_x: int,
    use_explain_position: bool,
    score_batch_size: int,
    channels_last: bool,
    attack_original_model: bool,
    verbose: bool,
) -> Tuple[Tensor, Tensor, List[int], List[int], List[List[Tuple[int, float, float, float]]]]:
    batch_size = x_rgb.shape[0]
    patch_positions: List[List[Tuple[int, int]]] = []
    for image_idx in range(batch_size):
        target_class = int(target_classes[image_idx].item())
        secondary_class = int(secondary_classes[image_idx].item())
        primary_guidance = guided.extract_attribution(
            guide_model,
            guided.to_bcos_input(x_rgb[image_idx:image_idx + 1]),
            target_class=target_class,
        )
        secondary_guidance = guided.extract_attribution(
            guide_model,
            guided.to_bcos_input(x_rgb[image_idx:image_idx + 1]),
            target_class=secondary_class,
        )
        patch_positions.append(
            guided.resolve_patch_positions(
                primary_guidance["contribution_map"],
                secondary_guidance["contribution_map"],
                patch_size,
                pos_y=pos_y,
                pos_x=pos_x,
                use_explain_position=use_explain_position,
            )
        )

    es = BatchedCirclePerturbPatchES(
        batch_size=batch_size,
        num_circles=num_circles,
        population_size=population_size,
        elite_fraction=elite_fraction,
        sigma=sigma,
        sigma_decay=sigma_decay,
        sigma_min=sigma_min,
        learning_rate=learning_rate,
        device=x_rgb.device,
    )

    histories: List[List[Tuple[int, float, float, float]]] = [[] for _ in range(batch_size)]
    best_score_overall = torch.full((batch_size,), -float("inf"), device=x_rgb.device)
    best_circles_overall = es.get_circles()
    best_pos_y_overall = [positions[0][0] for positions in patch_positions]
    best_pos_x_overall = [positions[0][1] for positions in patch_positions]

    for gen in range(generations):
        candidate_circles = es.ask()
        scores, best_pos_idx = compute_scores_fixed_batched_multi_position(
            attack_model,
            x_rgb,
            candidate_circles,
            patch_positions,
            target_classes,
            patch_size=patch_size,
            min_radius=min_radius,
            max_radius=max_radius,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
            rgb_epsilon=rgb_epsilon,
            linf_epsilon=linf_epsilon,
            edge_softness=edge_softness,
            score_batch_size=score_batch_size,
            channels_last=channels_last,
            attack_original_model=attack_original_model,
        )
        es.tell(candidate_circles, scores)

        best_scores, best_idx = scores.max(dim=1)
        gather_idx = best_idx.view(batch_size, 1, 1, 1).expand(-1, 1, num_circles, 7)
        best_circles = candidate_circles.gather(1, gather_idx).squeeze(1)
        best_pos_choice = best_pos_idx.gather(1, best_idx.unsqueeze(1)).squeeze(1)
        best_pos_y = [
            patch_positions[image_idx][int(best_pos_choice[image_idx].item())][0]
            for image_idx in range(batch_size)
        ]
        best_pos_x = [
            patch_positions[image_idx][int(best_pos_choice[image_idx].item())][1]
            for image_idx in range(batch_size)
        ]
        best_patches = render_patches_batch(
            x_rgb,
            best_circles,
            best_pos_y,
            best_pos_x,
            patch_size=patch_size,
            min_radius=min_radius,
            max_radius=max_radius,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
            rgb_epsilon=rgb_epsilon,
            linf_epsilon=linf_epsilon,
            edge_softness=edge_softness,
        )
        best_x = paste_patches_batch(x_rgb, best_patches, best_pos_y, best_pos_x)
        with torch.inference_mode():
            out_best = guided.model_forward_attack(
                attack_model,
                best_x,
                channels_last=channels_last,
                attack_original_model=attack_original_model,
            )
            best_logits = out_best.gather(1, target_classes.unsqueeze(1)).squeeze(1)

        improved = best_scores > best_score_overall
        best_score_overall = torch.where(improved, best_scores, best_score_overall)
        best_circles_overall[improved] = best_circles[improved]
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
                    f"Score: {best_scores[0].item():7.3f} | sigma: {es.sigma:.4f}"
                )
            else:
                print(
                    f"  Gen {gen + 1:3d}/{generations} | "
                    f"Mean Target Logit: {best_logits.mean().item():7.3f} | "
                    f"Best Target Logit: {best_logits.min().item():7.3f} | "
                    f"Mean Score: {best_scores.mean().item():7.3f} | sigma: {es.sigma:.4f}"
                )

    final_patches = render_patches_batch(
        x_rgb,
        best_circles_overall,
        best_pos_y_overall,
        best_pos_x_overall,
        patch_size=patch_size,
        min_radius=min_radius,
        max_radius=max_radius,
        min_alpha=min_alpha,
        max_alpha=max_alpha,
        rgb_epsilon=rgb_epsilon,
        linf_epsilon=linf_epsilon,
        edge_softness=edge_softness,
    )
    return final_patches, best_circles_overall, best_pos_y_overall, best_pos_x_overall, histories


def build_image_output_dir(output_root: Path, image_path: str, image_index: Optional[int]) -> Path:
    image_file = Path(image_path)
    stem_parts: List[str] = []
    if image_index is not None:
        stem_parts.append(f"{image_index:03d}")
    if image_file.parent.name:
        stem_parts.append(image_file.parent.name)
    stem_parts.append(image_file.stem)
    safe_name = "_".join(part for part in stem_parts if part)
    return output_root / f"{safe_name}_explain_guided_circle_rgb"


def resolve_plot_save_path(save_arg: str, out_dir: Path) -> str:
    if save_arg == "explain_guided_es_patch_circle_rgb.png":
        return str(out_dir / "explain_guided_es_patch_circle_rgb.png")
    save_path = Path(save_arg)
    if save_path.is_absolute():
        return str(save_path)
    return str(out_dir / save_path.name)


def result_circle_json(circles: Tensor, args: argparse.Namespace) -> str:
    return circles_to_json(
        circles,
        patch_size=args.patch_size,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        min_alpha=args.min_alpha,
        max_alpha=args.max_alpha,
        rgb_epsilon=args.rgb_epsilon,
    )


def process_single_image(
    guide_model: torch.nn.Module,
    attack_model: torch.nn.Module,
    image_path: str,
    image_index: Optional[int],
    output_root: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    x_rgb, resolved_image_path = guided.load_attack_rgb_image(
        image_path,
        device=args.device_obj,
        attack_original_model=args.attack_original_model,
    )
    x_rgb = guided.maybe_channels_last(x_rgb, args.channels_last)
    out_dir = build_image_output_dir(output_root, resolved_image_path, image_index)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n> Loading image ... {resolved_image_path}")

    before_path = out_dir / "before.png"
    after_path = out_dir / "after.png"
    if args.save_images:
        import torchvision.utils as vutils

        vutils.save_image(x_rgb, str(before_path))
        print(f"  Saved original image to {before_path}")

    with torch.inference_mode():
        out_orig = guided.model_forward_attack(
            attack_model,
            x_rgb,
            channels_last=args.channels_last,
            attack_original_model=args.attack_original_model,
        )
        target_class = out_orig.argmax(1).item()
        secondary_class = guided.resolve_runner_up_classes(
            out_orig,
            torch.tensor([target_class], device=out_orig.device, dtype=torch.long),
        )[0].item()
        logit_orig = out_orig[0, target_class].item()

    print(f"  Original class: {target_class} (logit: {logit_orig:.3f})")
    print(f"  Runner-up class: {secondary_class}")

    print(f"\n> Running explain-guided circle RGB perturbation {args.optimizer} optimization ...")
    if args.optimizer == "camo-1p1":
        final_patch, final_circles, final_pos_y, final_pos_x, history = (
            run_explain_guided_camo_one_plus_one_attack(
                guide_model=guide_model,
                attack_model=attack_model,
                x_rgb=x_rgb,
                target_class=target_class,
                secondary_class=secondary_class,
                patch_size=args.patch_size,
                num_circles=args.num_circles,
                generations=args.generations,
                camo_mutation_rate=args.camo_mut,
                min_radius=args.min_radius,
                max_radius=args.max_radius,
                min_alpha=args.min_alpha,
                max_alpha=args.max_alpha,
                rgb_epsilon=args.rgb_epsilon,
                linf_epsilon=args.linf_epsilon,
                edge_softness=args.edge_softness,
                pos_y=0,
                pos_x=0,
                use_explain_position=True,
                channels_last=args.channels_last,
                attack_original_model=args.attack_original_model,
                verbose=args.verbose_generations,
            )
        )
    else:
        final_patch, final_circles, final_pos_y, final_pos_x, history = run_explain_guided_attack(
            guide_model=guide_model,
            attack_model=attack_model,
            x_rgb=x_rgb,
            target_class=target_class,
            secondary_class=secondary_class,
            patch_size=args.patch_size,
            num_circles=args.num_circles,
            generations=args.generations,
            population_size=args.population,
            elite_fraction=args.elite_fraction,
            sigma=args.sigma,
            sigma_decay=args.sigma_decay,
            sigma_min=args.sigma_min,
            learning_rate=args.lr,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            min_alpha=args.min_alpha,
            max_alpha=args.max_alpha,
            rgb_epsilon=args.rgb_epsilon,
            linf_epsilon=args.linf_epsilon,
            edge_softness=args.edge_softness,
            pos_y=0,
            pos_x=0,
            use_explain_position=True,
            score_batch_size=args.score_batch_size,
            channels_last=args.channels_last,
            attack_original_model=args.attack_original_model,
            verbose=args.verbose_generations,
        )
    print(f"  Final patch position: ({final_pos_y}, {final_pos_x})")

    x_pert_rgb = paste_patch(x_rgb, final_patch, final_pos_y, final_pos_x)
    if args.save_images:
        import torchvision.utils as vutils

        vutils.save_image(x_pert_rgb, str(after_path))
        print(f"  Saved perturbed image to {after_path}")

    with torch.inference_mode():
        out_pert = guided.model_forward_attack(
            attack_model,
            x_pert_rgb,
            channels_last=args.channels_last,
            attack_original_model=args.attack_original_model,
        )
        pred_class_pert = out_pert.argmax(1).item()
        logit_pert_tgt = out_pert[0, target_class].item()
        logit_pert_pred = out_pert[0, pred_class_pert].item()

    print("\n> Attack complete")
    print(f"  Target class logit: {logit_orig:.3f} -> {logit_pert_tgt:.3f}")
    if pred_class_pert != target_class:
        print(f"  Class changed to {pred_class_pert} (logit: {logit_pert_pred:.3f})")
    else:
        print("  Prediction unchanged.")

    if args.save_figure:
        attr_orig = guided.extract_attribution(guide_model, guided.to_bcos_input(x_rgb), target_class=target_class)
        attr_pert = guided.extract_attribution(guide_model, guided.to_bcos_input(x_pert_rgb), target_class=target_class)

        save_plot_path = resolve_plot_save_path(args.save, out_dir)
        guided.visualize_patch_results(
            x_orig_rgb=x_rgb,
            x_pert_rgb=x_pert_rgb,
            patch=patch_rgb_for_visualization(final_patch),
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
        "optimizer": args.optimizer,
        "num_circles": args.num_circles,
        "circles": result_circle_json(final_circles, args),
        "success": int(pred_class_pert != target_class),
    }


def process_image_batch(
    guide_model: torch.nn.Module,
    attack_model: torch.nn.Module,
    image_items: List[Tuple[Optional[int], str]],
    output_root: Path,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    x_rgb_batch, resolved_image_paths = guided.load_rgb_image_batch(
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

    print(f"\n> Loading image batch ... {len(image_items)} images")
    for batch_idx, ((_, _), resolved_path) in enumerate(zip(image_items, resolved_image_paths), start=1):
        print(f"  [{batch_idx:02d}/{len(image_items):02d}] {resolved_path}")

    if args.save_images:
        import torchvision.utils as vutils

        for image_idx, out_dir in enumerate(out_dirs):
            before_path = out_dir / "before.png"
            vutils.save_image(x_rgb_batch[image_idx:image_idx + 1], str(before_path))

    with torch.inference_mode():
        out_orig = guided.model_forward_attack(
            attack_model,
            x_rgb_batch,
            channels_last=args.channels_last,
            attack_original_model=args.attack_original_model,
        )
        target_classes = out_orig.argmax(dim=1)
        secondary_classes = guided.resolve_runner_up_classes(out_orig, target_classes)
        logit_orig = out_orig.gather(1, target_classes.unsqueeze(1)).squeeze(1)

    print(f"\n> Running batched explain-guided circle RGB perturbation {args.optimizer} optimization ...")
    if args.optimizer == "camo-1p1":
        final_patches, final_circles, final_pos_y, final_pos_x, histories = (
            run_explain_guided_camo_one_plus_one_attack_batch(
                guide_model=guide_model,
                attack_model=attack_model,
                x_rgb=x_rgb_batch,
                target_classes=target_classes,
                secondary_classes=secondary_classes,
                patch_size=args.patch_size,
                num_circles=args.num_circles,
                generations=args.generations,
                camo_mutation_rate=args.camo_mut,
                min_radius=args.min_radius,
                max_radius=args.max_radius,
                min_alpha=args.min_alpha,
                max_alpha=args.max_alpha,
                rgb_epsilon=args.rgb_epsilon,
                linf_epsilon=args.linf_epsilon,
                edge_softness=args.edge_softness,
                pos_y=0,
                pos_x=0,
                use_explain_position=True,
                channels_last=args.channels_last,
                attack_original_model=args.attack_original_model,
                verbose=args.verbose_generations,
            )
        )
    else:
        final_patches, final_circles, final_pos_y, final_pos_x, histories = run_explain_guided_attack_batch(
            guide_model=guide_model,
            attack_model=attack_model,
            x_rgb=x_rgb_batch,
            target_classes=target_classes,
            secondary_classes=secondary_classes,
            patch_size=args.patch_size,
            num_circles=args.num_circles,
            generations=args.generations,
            population_size=args.population,
            elite_fraction=args.elite_fraction,
            sigma=args.sigma,
            sigma_decay=args.sigma_decay,
            sigma_min=args.sigma_min,
            learning_rate=args.lr,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            min_alpha=args.min_alpha,
            max_alpha=args.max_alpha,
            rgb_epsilon=args.rgb_epsilon,
            linf_epsilon=args.linf_epsilon,
            edge_softness=args.edge_softness,
            pos_y=0,
            pos_x=0,
            use_explain_position=True,
            score_batch_size=args.score_batch_size,
            channels_last=args.channels_last,
            attack_original_model=args.attack_original_model,
            verbose=args.verbose_generations,
        )

    x_pert_batch = paste_patches_batch(x_rgb_batch, final_patches, final_pos_y, final_pos_x)
    with torch.inference_mode():
        out_pert = guided.model_forward_attack(
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
            attr_orig = guided.extract_attribution(
                guide_model,
                guided.to_bcos_input(x_rgb_batch[batch_idx - 1:batch_idx]),
                target_class=target_class,
            )
            attr_pert = guided.extract_attribution(
                guide_model,
                guided.to_bcos_input(x_pert_batch[batch_idx - 1:batch_idx]),
                target_class=target_class,
            )
            save_plot_path = resolve_plot_save_path(args.save, out_dir)
            guided.visualize_patch_results(
                x_orig_rgb=x_rgb_batch[batch_idx - 1:batch_idx],
                x_pert_rgb=x_pert_batch[batch_idx - 1:batch_idx],
                patch=patch_rgb_for_visualization(final_patches[batch_idx - 1]),
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
                "optimizer": args.optimizer,
                "num_circles": args.num_circles,
                "circles": result_circle_json(final_circles[batch_idx - 1], args),
                "success": int(pred_class_pert != target_class),
            }
        )

    return results


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.image and args.images_csv:
        parser.error("Use either --image or --images-csv, not both.")
    if not args.image and not args.images_csv:
        parser.error("Provide --image or --images-csv.")
    if args.patch_size <= 0:
        parser.error("--patch-size must be > 0.")
    if args.num_circles <= 0:
        parser.error("--num-circles must be > 0.")
    if args.generations < 0:
        parser.error("--generations must be >= 0.")
    if args.population <= 0:
        parser.error("--population must be > 0.")
    if not (0 < args.elite_fraction <= 1):
        parser.error("--elite-fraction must be in (0, 1].")
    if not (0 <= args.camo_mut <= 1):
        parser.error("--camo-mut must be in [0, 1].")
    if args.image_batch_size < 0:
        parser.error("--image-batch-size must be >= 0.")
    if args.score_batch_size < 0:
        parser.error("--score-batch-size must be >= 0.")
    if args.sigma < 0 or args.sigma_min < 0:
        parser.error("--sigma and --sigma-min must be >= 0.")
    if args.sigma_decay <= 0:
        parser.error("--sigma-decay must be > 0.")
    if not (0 <= args.lr <= 1):
        parser.error("--lr must be in [0, 1].")
    if args.min_radius < 0 or args.max_radius < 0:
        parser.error("--min-radius and --max-radius must be >= 0.")
    if args.max_radius < args.min_radius:
        parser.error("--max-radius must be >= --min-radius.")
    if args.min_alpha < 0 or args.max_alpha < 0:
        parser.error("--min-alpha and --max-alpha must be >= 0.")
    if args.max_alpha < args.min_alpha:
        parser.error("--max-alpha must be >= --min-alpha.")
    if args.max_alpha > 1:
        parser.error("--max-alpha must be <= 1 because alpha is opacity.")
    if args.rgb_epsilon < 0:
        parser.error("--rgb-epsilon must be >= 0.")
    if args.linf_epsilon < 0:
        parser.error("--linf-epsilon must be >= 0.")
    if args.edge_softness < 0:
        parser.error("--edge-softness must be >= 0.")


def main() -> None:
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser(
        description="Explain-guided ES patch attack with circle RGB perturbation parameters",
    )
    parser.add_argument("--image", type=str, default=None, help="Path to one input image")
    parser.add_argument("--images-csv", type=str, default=None, help="CSV containing an image_path column")
    parser.add_argument("--model", type=str, default="resnet50", help="B-cos model name")
    parser.add_argument("--patch-size", type=int, default=16, help="Square patch size")
    parser.add_argument("--num-circles", type=int, default=8, help="Number of evolved perturbation circles per patch")
    parser.add_argument("--generations", type=int, default=200, help="Number of ES generations/queries")
    parser.add_argument(
        "--optimizer",
        choices=("camo-1p1", "population"),
        default="camo-1p1",
        help="ES optimizer: CamoPatch-style sequential (1+1), or the previous population elite update.",
    )
    parser.add_argument(
        "--camo-mut",
        "--mut",
        dest="camo_mut",
        type=float,
        default=0.3,
        help="CamoPatch-style mutation probability for --optimizer camo-1p1.",
    )
    parser.add_argument("--population", type=int, default=50, help="Population size")
    parser.add_argument("--elite-fraction", type=float, default=0.2, help="Elite fraction")
    parser.add_argument("--sigma", type=float, default=0.15, help="Initial normalized circle-parameter noise std")
    parser.add_argument("--sigma-decay", type=float, default=0.995, help="Sigma decay")
    parser.add_argument("--sigma-min", type=float, default=0.01, help="Minimum sigma")
    parser.add_argument("--lr", type=float, default=0.8, help="ES update rate")
    parser.add_argument("--min-radius", type=float, default=1.0, help="Minimum circle radius in patch pixels")
    parser.add_argument("--max-radius", type=float, default=8.0, help="Maximum circle radius in patch pixels")
    parser.add_argument("--min-alpha", type=float, default=0.0, help="Minimum circle opacity")
    parser.add_argument("--max-alpha", type=float, default=1.0, help="Maximum circle opacity")
    parser.add_argument("--rgb-epsilon", type=float, default=0.3, help="Maximum absolute RGB perturbation per circle")
    parser.add_argument("--linf-epsilon", type=float, default=0.3, help="Strict Linf budget for the final patch crop")
    parser.add_argument("--edge-softness", type=float, default=1.0, help="Circle edge feathering in pixels")
    parser.add_argument(
        "--score-batch-size",
        type=int,
        default=0,
        help="Total candidate evaluations per score forward. 0 = auto-chunk.",
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
    parser.add_argument("--save-images", action="store_true", help="Save before/after PNGs for each image")
    parser.add_argument("--save-figure", action="store_true", help="Render and save the explanation figure for each image")
    parser.add_argument("--verbose-generations", action="store_true", help="Print one log line per ES generation/query")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULT_DIR / "explain_guided_circle_rgb_outputs"),
        help="Root folder for all generated outputs",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="explain_guided_es_patch_circle_rgb.png",
        help="Output plot filename or path",
    )
    args = parser.parse_args()
    validate_args(parser, args)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    guided.configure_fast_runtime(device)
    args.device_obj = device
    args.channels_last = device.type == "cuda"
    if args.image_batch_size == 0:
        args.image_batch_size = 8 if args.images_csv and device.type == "cuda" else 1

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Explain-guided Circle RGB Perturbation ES Patch Attack")
    print("=" * 72)
    print(f"  Device       : {device}")
    print(f"  Model        : {args.model}")
    print(f"  Patch size   : {args.patch_size}x{args.patch_size}")
    print(f"  Circles      : {args.num_circles}")
    print(f"  Optimizer    : {args.optimizer}")
    if args.optimizer == "camo-1p1":
        print(f"  Camo mut     : {args.camo_mut:g}")
        print(f"  Queries      : {max(args.generations, 1)}")
    else:
        print(f"  Generations  : {args.generations}")
        print(f"  Population   : {args.population}")
    print(f"  Radius range : [{args.min_radius:g}, {args.max_radius:g}]")
    print(f"  Alpha range  : [{args.min_alpha:g}, {args.max_alpha:g}]")
    print(f"  RGB epsilon  : {args.rgb_epsilon:g}")
    print(f"  Linf budget  : {args.linf_epsilon:g}")
    print(f"  Attack mode  : {'torchvision-original' if args.attack_original_model else 'bcos'}")
    print("  Position rule: max patch sum(top1 contribution - top2 contribution)")
    if args.attack_original_model:
        print("  Guide only   : B-cos explain chooses position; ES optimize/eval run on torchvision model")
    print(f"  Image batch  : {args.image_batch_size}")
    print(f"  Score batch  : {'auto' if args.score_batch_size == 0 else args.score_batch_size}")
    print(f"  Save images  : {args.save_images}")
    print(f"  Save figure  : {args.save_figure}")
    print(f"  Output root  : {output_root}")

    print("\n> Loading B-cos guide model ...")
    model_fn = getattr(guided.bcos.pretrained, args.model)
    guide_model = model_fn(pretrained=True).to(device).eval()
    if args.channels_last:
        guide_model = guide_model.to(memory_format=torch.channels_last)

    if args.attack_original_model:
        print("> Loading original attack model ...")
        attack_model, attack_model_name, guided.ORIGINAL_ATTACK_PREPROCESS = guided.load_original_attack_model(
            args.model,
            device=device,
            channels_last=args.channels_last,
        )
        print(f"  Using torchvision model: {attack_model_name}")
        print(f"  Original input: Resize({IMAGENET_SL}) -> CenterCrop({IMAGENET_SL}) -> ToTensor()")
        print("  Normalize     : applied after perturbation inside model_forward_original")
    else:
        guided.ORIGINAL_ATTACK_PREPROCESS = None
        attack_model = guide_model

    if args.images_csv:
        image_items = guided.load_image_paths_from_csv(Path(args.images_csv))
        print(f"\n> Loaded {len(image_items)} image paths from {args.images_csv}")
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
            "optimizer",
            "num_circles",
            "circles",
            "success",
        ]
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        successes = sum(row["success"] for row in results)
        print(f"\n> Summary saved to {summary_path}")
        print(f"  Successful class changes: {successes}/{len(results)}")

    print("\n" + "=" * 72)
    print("  Done")
    print("=" * 72)


if __name__ == "__main__":
    main()
