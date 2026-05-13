"""
Batch comparison between built-in B-cos explanations and whitebox W1->L explanations.

For each image in a CSV, this script can perturb the input to change the
whitebox W1->L explanation while keeping the original predicted class. It then
reports Spearman rank correlation between the before/after explain-image
component. By default the component is the RGBA alpha channel.

The output also keeps `method_spearman`, the correlation between the built-in
B-cos explanation and the whitebox explanation on the final image.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from whitebox_w1l_attack import (
    fast_forward_matrix_linear_weights,
    linear_weights_to_image_tensor,
    load_rgb_input,
    save_explain_map,
    save_rgb_image,
    setup_sys_path,
    to_bcos_input,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT
DEFAULT_CSV_PATH = SCRIPT_DIR / "data" / "used_images_500.csv"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "artifacts" / "outputs" / "result" / "whitebox_explain_spearman_alpha.csv"
DEFAULT_ARTIFACTS_DIR = SCRIPT_DIR / "artifacts" / "outputs" / "result" / "whitebox_explain_spearman_artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Change whitebox W1->L explanations without changing the predicted "
            "class, then measure before/after Spearman rank correlation."
        ),
    )
    parser.add_argument(
        "--images-csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="CSV containing an image_path column.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Where to write per-image Spearman results.",
    )
    parser.add_argument("--model", default="resnet18", help="B-cos pretrained model entrypoint.")
    parser.add_argument("--limit", type=int, default=500, help="Maximum number of images to process.")
    parser.add_argument("--start", type=int, default=0, help="Skip this many valid CSV rows before processing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Class index to explain for every image. Defaults to each image's predicted class.",
    )
    parser.add_argument(
        "--target-column",
        default=None,
        help=(
            "Optional CSV column containing class indices to explain. Ignored when "
            "--target-class is provided. If omitted, the predicted class is used."
        ),
    )
    parser.add_argument(
        "--explain-smooth",
        type=int,
        default=15,
        help="Smoothing kernel used when rendering explanations.",
    )
    parser.add_argument(
        "--alpha-percentiles",
        type=float,
        nargs="+",
        default=[99.5],
        help="One or more alpha percentile cutoffs used when rendering explanations.",
    )
    parser.add_argument(
        "--component",
        choices=("alpha", "rgb", "rgba", "luma"),
        default="alpha",
        help="Explain-image component to flatten before Spearman correlation.",
    )
    parser.add_argument(
        "--attack-steps",
        type=int,
        default=0,
        help="If > 0, run Linf PGD to change the explanation while preserving the original class.",
    )
    parser.add_argument(
        "--attack-epsilon",
        type=float,
        default=8 / 255,
        help="Linf attack budget (used when --attack-steps > 0).",
    )
    parser.add_argument(
        "--attack-step-size",
        type=float,
        default=2 / 255,
        help="PGD step size (used when --attack-steps > 0).",
    )
    parser.add_argument(
        "--class-keep-weight",
        type=float,
        default=10.0,
        help="Penalty weight for keeping the original predicted class during explanation attack.",
    )
    parser.add_argument(
        "--class-keep-margin",
        type=float,
        default=0.0,
        help="Required logit margin for the original class during explanation attack.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directory to save adversarial image and explain maps.",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Save attack image and explain-before/after PNGs (when attack is enabled).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on the first image error instead of writing an error row and continuing.",
    )
    parser.add_argument(
        "--fast-runtime",
        action="store_true",
        help="Enable faster CUDA settings instead of stricter reproducibility settings.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print one line for every image.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def configure_runtime(device: torch.device, fast_runtime: bool) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = fast_runtime
    torch.backends.cudnn.allow_tf32 = fast_runtime
    torch.backends.cudnn.benchmark = fast_runtime
    if hasattr(torch.backends.cudnn, "deterministic"):
        torch.backends.cudnn.deterministic = not fast_runtime
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if fast_runtime else "highest")
    if hasattr(torch, "use_deterministic_algorithms") and not fast_runtime:
        torch.use_deterministic_algorithms(True, warn_only=True)


def load_image_items_from_csv(
    csv_path: Path,
    target_column: Optional[str],
    start: int,
    limit: int,
) -> List[Tuple[int, str, Optional[int]]]:
    if start < 0:
        raise ValueError("--start must be >= 0.")
    if limit <= 0:
        raise ValueError("--limit must be > 0.")

    items: List[Tuple[int, str, Optional[int]]] = []
    valid_row_index = 0
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        if "image_path" not in fieldnames:
            raise ValueError(f"CSV {csv_path} must contain an 'image_path' column.")
        if target_column is not None and target_column not in fieldnames:
            raise ValueError(f"CSV {csv_path} does not contain target column {target_column!r}.")

        for row_num, row in enumerate(reader, start=1):
            image_path = (row.get("image_path") or "").strip()
            if not image_path:
                continue
            valid_row_index += 1
            if valid_row_index <= start:
                continue

            target_value: Optional[int] = None
            if target_column is not None:
                raw_target = (row.get(target_column) or "").strip()
                if raw_target:
                    target_value = int(raw_target)

            raw_index = (row.get("index") or "").strip()
            image_index = int(raw_index) if raw_index.isdigit() else row_num
            items.append((image_index, image_path, target_value))
            if len(items) >= limit:
                break

    if not items:
        raise ValueError(f"CSV {csv_path} does not contain any image rows in the requested range.")
    return items


def rankdata_average(values: np.ndarray) -> np.ndarray:
    """Return average ranks for a 1D array, matching scipy.stats.rankdata(method='average')."""
    values = np.asarray(values)
    if values.ndim != 1:
        values = values.reshape(-1)
    sorter = np.argsort(values, kind="mergesort")
    sorted_values = values[sorter]

    new_group = np.empty(sorted_values.shape[0], dtype=bool)
    new_group[0] = True
    new_group[1:] = sorted_values[1:] != sorted_values[:-1]

    group_ids = np.cumsum(new_group) - 1
    counts = np.bincount(group_ids)
    ends = np.cumsum(counts)
    starts = ends - counts
    average_ranks = (starts + ends - 1).astype(np.float64) / 2.0

    ranks_sorted = average_ranks[group_ids]
    ranks = np.empty_like(ranks_sorted)
    ranks[sorter] = ranks_sorted
    return ranks


def spearman_rank_correlation(first: np.ndarray, second: np.ndarray) -> Tuple[float, int]:
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"Spearman inputs must have the same shape, got {a.shape} and {b.shape}.")

    finite = np.isfinite(a) & np.isfinite(b)
    a = a[finite]
    b = b[finite]
    n = int(a.size)
    if n < 2:
        return math.nan, n
    if np.all(a == a[0]) or np.all(b == b[0]):
        return math.nan, n

    rank_a = rankdata_average(a)
    rank_b = rankdata_average(b)
    rank_a = rank_a - rank_a.mean()
    rank_b = rank_b - rank_b.mean()
    denom = np.linalg.norm(rank_a) * np.linalg.norm(rank_b)
    if denom == 0:
        return math.nan, n
    return float(np.dot(rank_a, rank_b) / denom), n


def select_component(explanation: np.ndarray, component: str) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(explanation, dtype=np.float32))
    if arr.ndim != 3 or arr.shape[-1] != 4:
        raise ValueError(f"Expected RGBA explanation with shape [H, W, 4], got {arr.shape}.")

    if component == "alpha":
        return arr[..., 3]
    if component == "rgb":
        return arr[..., :3]
    if component == "rgba":
        return arr
    if component == "luma":
        rgb = arr[..., :3]
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    raise AssertionError(f"Unhandled component {component!r}.")


def select_component_tensor(explanation: Tensor, component: str) -> Tensor:
    if explanation.ndim != 4 or explanation.shape[1] != 4:
        raise ValueError(f"Expected RGBA explanation with shape [B, 4, H, W], got {tuple(explanation.shape)}.")

    if component == "alpha":
        return explanation[:, 3:4]
    if component == "rgb":
        return explanation[:, :3]
    if component == "rgba":
        return explanation
    if component == "luma":
        rgb = explanation[:, :3]
        return 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    raise AssertionError(f"Unhandled component {component!r}.")


def class_margin(logits: Tensor, class_idx: int) -> Tensor:
    class_logit = logits[:, class_idx]
    other_logits = logits.clone()
    other_logits[:, class_idx] = -float("inf")
    return class_logit - other_logits.max(dim=1).values


def compute_builtin_dynamic_weights(
    model: torch.nn.Module,
    x_bcos: Tensor,
    target_class: int,
) -> Tensor:
    inp = x_bcos.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    result = model.explain(inp, idx=target_class)
    return result["dynamic_linear_weights"].detach().clone()


def render_builtin_explanation(
    x_bcos: Tensor,
    dynamic_linear_weights: Tensor,
    smooth: int,
    alpha_percentile: float,
) -> np.ndarray:
    from bcos.common import gradient_to_image

    return np.nan_to_num(
        gradient_to_image(
            x_bcos[0],
            dynamic_linear_weights[0],
            smooth=smooth,
            alpha_percentile=alpha_percentile,
        ),
    ).astype(np.float32)


def render_whitebox_explanation(
    x_bcos: Tensor,
    linear_weights: Tensor,
    smooth: int,
    alpha_percentile: float,
) -> np.ndarray:
    rgba = linear_weights_to_image_tensor(
        x_bcos,
        linear_weights,
        smooth=smooth,
        alpha_percentile=alpha_percentile,
    )
    return np.nan_to_num(rgba[0].detach().cpu().permute(1, 2, 0).numpy()).astype(np.float32)


def sanitize_stem(text: str) -> str:
    stem = Path(text).stem
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return clean or "image"


def pgd_change_explanation_keep_class_linf(
    model: torch.nn.Module,
    add_inverse: Any,
    x_rgb: Tensor,
    explain_class: int,
    keep_class: int,
    clean_component: Tensor,
    component_name: str,
    smooth: int,
    alpha_percentile: float,
    epsilon: float,
    step_size: float,
    steps: int,
    class_keep_weight: float,
    class_keep_margin: float,
) -> Tuple[Tensor, float, float]:
    """Linf PGD that changes the whitebox explanation while keeping keep_class."""
    if steps <= 0:
        with torch.inference_mode():
            logits = model(to_bcos_input(x_rgb, add_inverse=add_inverse))
            margin = float(class_margin(logits, keep_class)[0].item())
        return x_rgb.detach().clone(), 0.0, margin

    delta = torch.empty_like(x_rgb).uniform_(-epsilon, epsilon)
    delta = ((x_rgb + delta).clamp(0.0, 1.0) - x_rgb).detach()
    delta.requires_grad_(True)
    best_delta = torch.zeros_like(x_rgb)
    best_change = 0.0
    with torch.inference_mode():
        clean_logits = model(to_bcos_input(x_rgb, add_inverse=add_inverse))
        best_margin = float(class_margin(clean_logits, keep_class)[0].item())

    for _ in range(steps):
        x_adv = (x_rgb + delta).clamp(0.0, 1.0)
        x_adv_bcos = to_bcos_input(x_adv, add_inverse=add_inverse)
        logits = model(x_adv_bcos)
        margin = class_margin(logits, keep_class)

        _, weights = fast_forward_matrix_linear_weights(
            model=model,
            x_bcos=x_adv_bcos,
            target_class=explain_class,
            create_graph=True,
        )
        rendered = linear_weights_to_image_tensor(
            x_adv_bcos,
            weights,
            smooth=smooth,
            alpha_percentile=alpha_percentile,
        )
        component = select_component_tensor(rendered, component_name)
        component_flat = component.flatten(start_dim=1)
        clean_component_flat = clean_component.flatten(start_dim=1)
        cosine_similarity = F.cosine_similarity(
            component_flat,
            clean_component_flat,
            dim=1,
            eps=1e-8,
        )
        explain_change = (1.0 - cosine_similarity).mean()
        class_penalty = F.relu(margin.new_tensor(class_keep_margin) - margin).pow(2).mean()
        objective = explain_change - class_keep_weight * class_penalty

        grad = torch.autograd.grad(objective, delta, retain_graph=False, create_graph=False)[0]
        with torch.no_grad():
            pred_class = int(logits.argmax(dim=1).item())
            change_value = float(explain_change.detach().item())
            margin_value = float(margin.detach()[0].item())
            if pred_class == keep_class and change_value >= best_change:
                best_change = change_value
                best_margin = margin_value
                best_delta = delta.detach().clone()

            delta.add_(step_size * grad.sign())
            delta.clamp_(-epsilon, epsilon)
            adv = (x_rgb + delta).clamp(0.0, 1.0)
            delta.copy_(adv - x_rgb)
        delta.grad = None

    return (x_rgb + best_delta).clamp(0.0, 1.0).detach(), best_change, best_margin


def save_attack_artifacts(
    args: argparse.Namespace,
    image_index: int,
    image_path: str,
    alpha_percentile: float,
    x_orig_rgb: Tensor,
    x_adv_rgb: Tensor,
    explain_before_builtin: np.ndarray,
    explain_before_whitebox: np.ndarray,
    explain_after_builtin: np.ndarray,
    explain_after_whitebox: np.ndarray,
) -> Dict[str, str]:
    item_name = sanitize_stem(image_path)
    alpha_tag = str(alpha_percentile).replace(".", "p")
    out_dir = args.artifacts_dir / f"{image_index:05d}_{item_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_path = out_dir / "image_before_attack.png"
    adv_path = out_dir / "image_after_attack.png"
    before_builtin_path = out_dir / f"explain_before_builtin_a{alpha_tag}.png"
    before_whitebox_path = out_dir / f"explain_before_whitebox_a{alpha_tag}.png"
    after_builtin_path = out_dir / f"explain_after_builtin_a{alpha_tag}.png"
    after_whitebox_path = out_dir / f"explain_after_whitebox_a{alpha_tag}.png"

    save_rgb_image(x_orig_rgb, orig_path)
    save_rgb_image(x_adv_rgb, adv_path)
    save_explain_map(explain_before_builtin, before_builtin_path)
    save_explain_map(explain_before_whitebox, before_whitebox_path)
    save_explain_map(explain_after_builtin, after_builtin_path)
    save_explain_map(explain_after_whitebox, after_whitebox_path)

    return {
        "orig_image_path": str(orig_path),
        "adv_image_path": str(adv_path),
        "explain_before_builtin_path": str(before_builtin_path),
        "explain_before_whitebox_path": str(before_whitebox_path),
        "explain_after_builtin_path": str(after_builtin_path),
        "explain_after_whitebox_path": str(after_whitebox_path),
    }


def compute_image_rows(
    model: torch.nn.Module,
    add_inverse: Any,
    image_index: int,
    image_path: str,
    csv_target_class: Optional[int],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    x_rgb = load_rgb_input(image_path, model, device=args.device_obj)
    x_bcos = to_bcos_input(x_rgb, add_inverse=add_inverse).detach()

    with torch.inference_mode():
        outputs = model(x_bcos)
        pred_class = int(outputs.argmax(dim=1).item())
        pred_logit = float(outputs[0, pred_class].item())

    if args.target_class is not None:
        target_class = int(args.target_class)
        target_source = "arg"
    elif csv_target_class is not None:
        target_class = int(csv_target_class)
        target_source = "csv"
    else:
        target_class = pred_class
        target_source = "prediction"
    target_logit = float(outputs[0, target_class].item())

    clean_builtin_weights = compute_builtin_dynamic_weights(
        model=model,
        x_bcos=x_bcos,
        target_class=target_class,
    )
    _, clean_whitebox_weights = fast_forward_matrix_linear_weights(
        model=model,
        x_bcos=x_bcos,
        target_class=target_class,
        create_graph=False,
    )

    attack_enabled = args.attack_steps > 0
    x_adv_rgb = x_rgb
    attack_explain_cosine_distance = 0.0
    attack_best_margin = float(class_margin(outputs, pred_class)[0].item())
    if attack_enabled:
        attack_alpha = args.alpha_percentiles[0]
        clean_attack_explain = linear_weights_to_image_tensor(
            x_bcos,
            clean_whitebox_weights,
            smooth=args.explain_smooth,
            alpha_percentile=attack_alpha,
        )
        clean_attack_component = select_component_tensor(
            clean_attack_explain,
            args.component,
        ).detach()
        (
            x_adv_rgb,
            attack_explain_cosine_distance,
            attack_best_margin,
        ) = pgd_change_explanation_keep_class_linf(
            model=model,
            add_inverse=add_inverse,
            x_rgb=x_rgb,
            explain_class=target_class,
            keep_class=pred_class,
            clean_component=clean_attack_component,
            component_name=args.component,
            smooth=args.explain_smooth,
            alpha_percentile=attack_alpha,
            epsilon=args.attack_epsilon,
            step_size=args.attack_step_size,
            steps=args.attack_steps,
            class_keep_weight=args.class_keep_weight,
            class_keep_margin=args.class_keep_margin,
        )

    with torch.inference_mode():
        adv_outputs = model(to_bcos_input(x_adv_rgb, add_inverse=add_inverse))
        adv_pred_class = int(adv_outputs.argmax(dim=1).item())
        adv_target_logit = float(adv_outputs[0, target_class].item())
        adv_keep_logit = float(adv_outputs[0, pred_class].item())
        adv_keep_margin = float(class_margin(adv_outputs, pred_class)[0].item())
        class_preserved = int(adv_pred_class == pred_class)

    x_adv_bcos = to_bcos_input(x_adv_rgb, add_inverse=add_inverse).detach()

    builtin_weights = compute_builtin_dynamic_weights(
        model=model,
        x_bcos=x_adv_bcos,
        target_class=target_class,
    )
    _, whitebox_weights = fast_forward_matrix_linear_weights(
        model=model,
        x_bcos=x_adv_bcos,
        target_class=target_class,
        create_graph=False,
    )

    rows: List[Dict[str, Any]] = []
    artifact_paths: Dict[str, str] = {
        "orig_image_path": "",
        "adv_image_path": "",
        "explain_before_builtin_path": "",
        "explain_before_whitebox_path": "",
        "explain_after_builtin_path": "",
        "explain_after_whitebox_path": "",
    }

    if args.save_artifacts:
        alpha_for_save = args.alpha_percentiles[0]
        explain_before_builtin = render_builtin_explanation(
            x_bcos=x_bcos,
            dynamic_linear_weights=clean_builtin_weights,
            smooth=args.explain_smooth,
            alpha_percentile=alpha_for_save,
        )
        explain_before_whitebox = render_whitebox_explanation(
            x_bcos=x_bcos,
            linear_weights=clean_whitebox_weights,
            smooth=args.explain_smooth,
            alpha_percentile=alpha_for_save,
        )
        explain_after_builtin = render_builtin_explanation(
            x_bcos=x_adv_bcos,
            dynamic_linear_weights=builtin_weights,
            smooth=args.explain_smooth,
            alpha_percentile=alpha_for_save,
        )
        explain_after_whitebox = render_whitebox_explanation(
            x_bcos=x_adv_bcos,
            linear_weights=whitebox_weights,
            smooth=args.explain_smooth,
            alpha_percentile=alpha_for_save,
        )
        artifact_paths = save_attack_artifacts(
            args=args,
            image_index=image_index,
            image_path=image_path,
            alpha_percentile=alpha_for_save,
            x_orig_rgb=x_rgb,
            x_adv_rgb=x_adv_rgb,
            explain_before_builtin=explain_before_builtin,
            explain_before_whitebox=explain_before_whitebox,
            explain_after_builtin=explain_after_builtin,
            explain_after_whitebox=explain_after_whitebox,
        )

    for alpha_percentile in args.alpha_percentiles:
        clean_builtin_explain = render_builtin_explanation(
            x_bcos=x_bcos,
            dynamic_linear_weights=clean_builtin_weights,
            smooth=args.explain_smooth,
            alpha_percentile=alpha_percentile,
        )
        clean_whitebox_explain = render_whitebox_explanation(
            x_bcos=x_bcos,
            linear_weights=clean_whitebox_weights,
            smooth=args.explain_smooth,
            alpha_percentile=alpha_percentile,
        )
        adv_builtin_explain = render_builtin_explanation(
            x_bcos=x_adv_bcos,
            dynamic_linear_weights=builtin_weights,
            smooth=args.explain_smooth,
            alpha_percentile=alpha_percentile,
        )
        adv_whitebox_explain = render_whitebox_explanation(
            x_bcos=x_adv_bcos,
            linear_weights=whitebox_weights,
            smooth=args.explain_smooth,
            alpha_percentile=alpha_percentile,
        )

        clean_builtin_component = select_component(clean_builtin_explain, args.component)
        clean_whitebox_component = select_component(clean_whitebox_explain, args.component)
        adv_builtin_component = select_component(adv_builtin_explain, args.component)
        adv_whitebox_component = select_component(adv_whitebox_explain, args.component)

        spearman, n_values = spearman_rank_correlation(clean_whitebox_component, adv_whitebox_component)
        builtin_before_after_spearman, _ = spearman_rank_correlation(
            clean_builtin_component,
            adv_builtin_component,
        )
        method_spearman, _ = spearman_rank_correlation(adv_builtin_component, adv_whitebox_component)
        mean_abs_diff = float(np.mean(np.abs(clean_whitebox_component - adv_whitebox_component)))
        max_abs_diff = float(np.max(np.abs(clean_whitebox_component - adv_whitebox_component)))
        weight_max_abs_diff = float((builtin_weights - whitebox_weights).abs().max().item())

        rows.append(
            {
                "index": image_index,
                "image_path": image_path,
                "model": args.model,
                "target_class": target_class,
                "target_source": target_source,
                "pred_class": pred_class,
                "pred_logit": f"{pred_logit:.8g}",
                "target_logit": f"{target_logit:.8g}",
                "attack_enabled": str(attack_enabled),
                "attack_steps": args.attack_steps,
                "attack_epsilon": f"{args.attack_epsilon:.10g}",
                "attack_step_size": f"{args.attack_step_size:.10g}",
                "adv_pred_class": adv_pred_class,
                "adv_target_logit": f"{adv_target_logit:.8g}",
                "adv_keep_logit": f"{adv_keep_logit:.8g}",
                "adv_keep_margin": f"{adv_keep_margin:.8g}",
                "class_preserved": class_preserved,
                "attack_explain_cosine_distance": f"{attack_explain_cosine_distance:.10g}",
                "attack_best_margin": f"{attack_best_margin:.8g}",
                "orig_image_path": artifact_paths["orig_image_path"],
                "adv_image_path": artifact_paths["adv_image_path"],
                "explain_before_builtin_path": artifact_paths["explain_before_builtin_path"],
                "explain_before_whitebox_path": artifact_paths["explain_before_whitebox_path"],
                "explain_after_builtin_path": artifact_paths["explain_after_builtin_path"],
                "explain_after_whitebox_path": artifact_paths["explain_after_whitebox_path"],
                "component": args.component,
                "smooth": args.explain_smooth,
                "alpha_percentile": f"{alpha_percentile:g}",
                "spearman": f"{spearman:.10g}" if math.isfinite(spearman) else "nan",
                "whitebox_before_after_spearman": f"{spearman:.10g}" if math.isfinite(spearman) else "nan",
                "builtin_before_after_spearman": (
                    f"{builtin_before_after_spearman:.10g}"
                    if math.isfinite(builtin_before_after_spearman)
                    else "nan"
                ),
                "method_spearman": f"{method_spearman:.10g}" if math.isfinite(method_spearman) else "nan",
                "n_values": n_values,
                "mean_abs_diff": f"{mean_abs_diff:.10g}",
                "max_abs_diff": f"{max_abs_diff:.10g}",
                "weights_max_abs_diff": f"{weight_max_abs_diff:.10g}",
                "status": "ok",
                "error": "",
            },
        )
    return rows


def error_rows(
    image_index: int,
    image_path: str,
    args: argparse.Namespace,
    error: Exception,
) -> List[Dict[str, Any]]:
    rows = []
    for alpha_percentile in args.alpha_percentiles:
        rows.append(
            {
                "index": image_index,
                "image_path": image_path,
                "model": args.model,
                "target_class": "",
                "target_source": "",
                "pred_class": "",
                "pred_logit": "",
                "target_logit": "",
                "attack_enabled": str(args.attack_steps > 0),
                "attack_steps": args.attack_steps,
                "attack_epsilon": f"{args.attack_epsilon:.10g}",
                "attack_step_size": f"{args.attack_step_size:.10g}",
                "adv_pred_class": "",
                "adv_target_logit": "",
                "adv_keep_logit": "",
                "adv_keep_margin": "",
                "class_preserved": "",
                "attack_explain_cosine_distance": "nan",
                "attack_best_margin": "nan",
                "orig_image_path": "",
                "adv_image_path": "",
                "explain_before_builtin_path": "",
                "explain_before_whitebox_path": "",
                "explain_after_builtin_path": "",
                "explain_after_whitebox_path": "",
                "component": args.component,
                "smooth": args.explain_smooth,
                "alpha_percentile": f"{alpha_percentile:g}",
                "spearman": "nan",
                "whitebox_before_after_spearman": "nan",
                "builtin_before_after_spearman": "nan",
                "method_spearman": "nan",
                "n_values": 0,
                "mean_abs_diff": "nan",
                "max_abs_diff": "nan",
                "weights_max_abs_diff": "nan",
                "status": "error",
                "error": str(error),
            },
        )
    return rows


def write_results_csv(output_csv: Path, rows: Sequence[Dict[str, Any]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "image_path",
        "model",
        "target_class",
        "target_source",
        "pred_class",
        "pred_logit",
        "target_logit",
        "attack_enabled",
        "attack_steps",
        "attack_epsilon",
        "attack_step_size",
        "adv_pred_class",
        "adv_target_logit",
        "adv_keep_logit",
        "adv_keep_margin",
        "class_preserved",
        "attack_explain_cosine_distance",
        "attack_best_margin",
        "orig_image_path",
        "adv_image_path",
        "explain_before_builtin_path",
        "explain_before_whitebox_path",
        "explain_after_builtin_path",
        "explain_after_whitebox_path",
        "component",
        "smooth",
        "alpha_percentile",
        "spearman",
        "whitebox_before_after_spearman",
        "builtin_before_after_spearman",
        "method_spearman",
        "n_values",
        "mean_abs_diff",
        "max_abs_diff",
        "weights_max_abs_diff",
        "status",
        "error",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float_field(row: Dict[str, Any], key: str) -> Optional[float]:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    values = [v for row in ok_rows if (v := parse_float_field(row, "spearman")) is not None]
    preserved_rows = sum(str(row.get("class_preserved")).lower() in {"1", "true"} for row in ok_rows)
    if not values:
        return {
            "ok_rows": len(ok_rows),
            "error_rows": len(rows) - len(ok_rows),
            "class_preserved_rows": preserved_rows,
            "count": 0,
        }

    return {
        "ok_rows": len(ok_rows),
        "error_rows": len(rows) - len(ok_rows),
        "class_preserved_rows": preserved_rows,
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def format_alpha_list(values: Iterable[float]) -> str:
    return ", ".join(f"{value:g}" for value in values)


def main() -> None:
    args = parse_args()
    setup_sys_path()
    import bcos
    from bcos import transforms as bcos_transforms

    if args.explain_smooth < 0:
        raise ValueError("--explain-smooth must be >= 0.")
    if args.attack_steps < 0:
        raise ValueError("--attack-steps must be >= 0.")
    if args.attack_epsilon < 0:
        raise ValueError("--attack-epsilon must be >= 0.")
    if args.attack_step_size < 0:
        raise ValueError("--attack-step-size must be >= 0.")
    if args.class_keep_weight < 0:
        raise ValueError("--class-keep-weight must be >= 0.")
    for value in args.alpha_percentiles:
        if not 0 <= value <= 100:
            raise ValueError("--alpha-percentiles values must be in [0, 100].")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    configure_runtime(device, fast_runtime=args.fast_runtime)
    args.device_obj = device

    image_items = load_image_items_from_csv(
        csv_path=args.images_csv,
        target_column=args.target_column,
        start=args.start,
        limit=args.limit,
    )

    print("=" * 72)
    print("  Whitebox Explain Spearman Experiment")
    print("=" * 72)
    print(f"  CSV             : {args.images_csv}")
    print(f"  Images          : {len(image_items)}")
    print(f"  Model           : {args.model}")
    print(f"  Device          : {device}")
    print(f"  Component       : {args.component}")
    print(f"  Smooth          : {args.explain_smooth}")
    print(f"  Alpha percentile: {format_alpha_list(args.alpha_percentiles)}")
    print(f"  Attack steps    : {args.attack_steps}")
    if args.attack_steps > 0:
        print(f"  Attack eps      : {args.attack_epsilon:g}")
        print(f"  Attack step size: {args.attack_step_size:g}")
        print(f"  Keep-class wgt  : {args.class_keep_weight:g}")
        print(f"  Keep margin     : {args.class_keep_margin:g}")
        print(f"  Save artifacts  : {args.save_artifacts}")
        if args.save_artifacts:
            print(f"  Artifacts dir   : {args.artifacts_dir}")
    print(f"  Output CSV      : {args.output_csv}")
    print()

    model_fn = getattr(bcos.pretrained, args.model)
    model = model_fn(pretrained=True).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    add_inverse = bcos_transforms.AddInverse()

    all_rows: List[Dict[str, Any]] = []
    start_time = time.perf_counter()
    for item_num, (image_index, image_path, csv_target_class) in enumerate(image_items, start=1):
        if args.verbose:
            print(f"[{item_num:04d}/{len(image_items):04d}] {image_path}")
        try:
            rows = compute_image_rows(
                model=model,
                add_inverse=add_inverse,
                image_index=image_index,
                image_path=image_path,
                csv_target_class=csv_target_class,
                args=args,
            )
        except Exception as exc:
            if args.strict:
                raise
            rows = error_rows(image_index, image_path, args, exc)
            print(f"  warning: failed image {image_index} ({image_path}): {exc}", file=sys.stderr)

        all_rows.extend(rows)
        if not args.verbose and item_num % 25 == 0:
            elapsed = time.perf_counter() - start_time
            print(f"  processed {item_num}/{len(image_items)} images in {elapsed:.1f}s")

    write_results_csv(args.output_csv, all_rows)
    summary = summarize(all_rows)
    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 72)
    print("  Summary")
    print("=" * 72)
    print(f"  Result rows     : {len(all_rows)}")
    print(f"  OK rows         : {summary['ok_rows']}")
    print(f"  Error rows      : {summary['error_rows']}")
    print(f"  Class preserved : {summary['class_preserved_rows']}/{summary['ok_rows']}")
    if summary["count"]:
        print(f"  Spearman count  : {summary['count']}")
        print(f"  Spearman mean   : {summary['mean']:.6f}")
        print(f"  Spearman median : {summary['median']:.6f}")
        print(f"  Spearman std    : {summary['stdev']:.6f}")
        print(f"  Spearman min/max: {summary['min']:.6f} / {summary['max']:.6f}")
    else:
        print("  Spearman count  : 0")
    print(f"  Elapsed         : {elapsed:.1f}s")
    print(f"  Saved           : {args.output_csv}")


if __name__ == "__main__":
    main()
