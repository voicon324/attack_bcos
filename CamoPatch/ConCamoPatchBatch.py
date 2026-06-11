from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from CamoPatch import (
    l2,
    linf,
    mutate,
    project_linf,
    render,
    save_rgb_image,
    sh_selection,
    update_location,
)
from ConCamoPatch import (
    describe_attack_model,
    parse_linf,
    pytorch_switch,
    resolve_bcos_guide_model,
)
from ImageNetModels import ImageNetModel


IMAGENET_SL = 224
POSITION_RULES = ("random", "margin", "top1", "dynamic-margin", "dynamic", "gradcam")


def bcos_location_source(position_rule: str, fixed_position: bool) -> str:
    suffix = "fixed" if fixed_position else "init"
    if position_rule == "margin":
        return f"bcos_contribution_margin_{suffix}"
    return f"bcos_{position_rule.replace('-', '_')}_{suffix}"


def extract_bcos_gradcam_map(
    guide_model: torch.nn.Module,
    x_bcos: torch.Tensor,
    target_class: int,
) -> torch.Tensor:
    try:
        features = guide_model.get_feature_extractor()
        classifier = guide_model.get_classifier()
    except AttributeError as exc:
        raise AttributeError(
            "Grad-CAM position rule requires a B-cos model with "
            "get_feature_extractor() and get_classifier()."
        ) from exc

    with torch.no_grad():
        feature_map = features(x_bcos)

    with torch.enable_grad():
        var_features = feature_map.detach().requires_grad_(True)
        logits = F.adaptive_avg_pool2d(classifier(var_features), 1)[..., 0, 0]
        target_score = logits[0, int(target_class)]
        grads = torch.autograd.grad(target_score, var_features)[0]
        gradcam = (grads.sum(dim=(-2, -1), keepdim=True) * var_features).sum(dim=1, keepdim=True)
        gradcam = gradcam.relu()
        return F.interpolate(gradcam, size=x_bcos.shape[-2:], mode="nearest")


def resolve_bcos_position_by_rule(
    model: ImageNetModel,
    x_rgb_chw: torch.Tensor,
    true_label: int,
    patch_size: int,
    model_idx: int,
    device: Optional[str],
    position_rule: str,
    guide: Optional[ImageNetModel] = None,
) -> Tuple[np.ndarray, int, int]:
    if position_rule == "random":
        raise ValueError("Random positions are resolved in the batch runner, not through B-cos.")

    repo_root = Path(__file__).resolve().parents[1]
    attacks_dir = repo_root / "attacks"
    import sys

    for path in (repo_root, attacks_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    from attacks.explain_guided_pixel_es_patch import (
        extract_attribution,
        find_best_patch_position_from_contribution_map,
        find_best_patch_position_from_contribution_margin,
        find_best_patch_position_from_dynamic_map,
        find_best_patch_position_from_dynamic_margin,
        resolve_runner_up_classes,
        to_bcos_input,
    )

    if guide is None:
        guide = resolve_bcos_guide_model(model, model_idx, device)

    guide_model = guide.model
    guide_device = guide.device
    x_rgb = x_rgb_chw.unsqueeze(0).to(device=guide_device, dtype=torch.float32)
    x_bcos = to_bcos_input(x_rgb)

    with torch.inference_mode():
        outputs = guide_model(x_bcos)
        target_classes = torch.tensor([int(true_label)], device=outputs.device, dtype=torch.long)
        secondary_class = int(resolve_runner_up_classes(outputs, target_classes)[0].item())
        guide_prediction = int(outputs.argmax(dim=1)[0].item())

    if position_rule == "gradcam":
        gradcam_map = extract_bcos_gradcam_map(guide_model, x_bcos, int(true_label))
        pos_y, pos_x = find_best_patch_position_from_contribution_map(gradcam_map, patch_size)
    else:
        primary_guidance = extract_attribution(guide_model, x_bcos, target_class=int(true_label))
        secondary_guidance = extract_attribution(guide_model, x_bcos, target_class=secondary_class)
        if position_rule == "margin":
            pos_y, pos_x = find_best_patch_position_from_contribution_margin(
                primary_guidance["contribution_map"],
                secondary_guidance["contribution_map"],
                patch_size,
            )
        elif position_rule == "top1":
            pos_y, pos_x = find_best_patch_position_from_contribution_map(
                primary_guidance["contribution_map"],
                patch_size,
            )
        elif position_rule == "dynamic-margin":
            pos_y, pos_x = find_best_patch_position_from_dynamic_margin(
                primary_guidance["dynamic_linear_weights"],
                secondary_guidance["dynamic_linear_weights"],
                patch_size,
            )
        elif position_rule == "dynamic":
            pos_y, pos_x = find_best_patch_position_from_dynamic_map(
                primary_guidance["dynamic_linear_weights"],
                patch_size,
            )
        else:
            raise ValueError(f"Unsupported position rule: {position_rule}")

    return np.array([pos_y, pos_x], dtype=np.int64), guide_prediction, secondary_class


def load_items_from_csv(csv_path: Path) -> List[Tuple[int, int, str, str]]:
    label_columns = ("true_label", "label", "class_idx", "class_id", "target_class", "pred_class")
    items: List[Tuple[int, int, str, str]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        image_column = "image_path" if "image_path" in reader.fieldnames else reader.fieldnames[0]
        label_column = next((name for name in label_columns if name in reader.fieldnames), None)

        for idx, row in enumerate(reader, start=1):
            image_path = (row.get(image_column) or "").strip()
            if not image_path:
                continue
            if label_column and (row.get(label_column) or "").strip():
                true_label = int(float(row[label_column]))
            else:
                parent = Path(image_path).parent.name
                if not parent.isdigit():
                    raise ValueError(
                        f"Cannot infer label for row {idx}: no label column and parent dir is {parent!r}"
                    )
                true_label = int(parent)
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(image_path).stem).strip("._")
            items.append((idx, true_label, stem, image_path))
    if not items:
        raise ValueError(f"CSV has no valid image rows: {csv_path}")
    return items


def load_image_batch(
    image_paths: List[str],
    load_image: transforms.Compose,
) -> Tuple[np.ndarray, List[torch.Tensor]]:
    chw_tensors: List[torch.Tensor] = []
    nhwc_arrays: List[np.ndarray] = []
    for image_path in image_paths:
        x_chw = load_image(Image.open(image_path).convert("RGB"))
        chw_tensors.append(x_chw)
        nhwc_arrays.append(pytorch_switch(x_chw).detach().numpy())
    return np.stack(nhwc_arrays, axis=0), chw_tensors


def render_patches_for_batch(
    x_batch: np.ndarray,
    patch_genos: np.ndarray,
    locs: np.ndarray,
    patch_size: int,
    eps_linf: Optional[float],
) -> np.ndarray:
    patches = np.stack([render(patch_genos[idx], patch_size) for idx in range(patch_genos.shape[0])], axis=0)
    if eps_linf is None:
        return patches
    projected: List[np.ndarray] = []
    for idx, patch in enumerate(patches):
        y, x = locs[idx]
        orig_patch = x_batch[idx, y:y + patch_size, x:x + patch_size, :]
        projected.append(project_linf(patch, orig_patch, eps_linf))
    return np.stack(projected, axis=0)


def apply_patches_to_batch(
    x_batch: np.ndarray,
    patches: np.ndarray,
    locs: np.ndarray,
) -> np.ndarray:
    x_adv = x_batch.copy()
    patch_size = patches.shape[1]
    for idx, patch in enumerate(patches):
        y, x = locs[idx]
        x_adv[idx, y:y + patch_size, x:x + patch_size, :] = patch
    np.clip(x_adv, 0.0, 1.0, out=x_adv)
    return x_adv


def patch_l2_batch(
    x_batch: np.ndarray,
    patches: np.ndarray,
    locs: np.ndarray,
) -> np.ndarray:
    values = np.empty((patches.shape[0],), dtype=np.float64)
    patch_size = patches.shape[1]
    for idx, patch in enumerate(patches):
        y, x = locs[idx]
        orig_patch = x_batch[idx, y:y + patch_size, x:x + patch_size, :]
        values[idx] = l2(patch, orig_patch)
    return values


def patch_linf_batch(
    x_batch: np.ndarray,
    patches: np.ndarray,
    locs: np.ndarray,
) -> np.ndarray:
    values = np.empty((patches.shape[0],), dtype=np.float64)
    patch_size = patches.shape[1]
    for idx, patch in enumerate(patches):
        y, x = locs[idx]
        orig_patch = x_batch[idx, y:y + patch_size, x:x + patch_size, :]
        values[idx] = linf(patch, orig_patch)
    return values


def evaluate_batch(
    model: ImageNetModel,
    x_batch: np.ndarray,
    true_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    outputs = model.predict(x_batch)
    if torch.is_tensor(outputs):
        preds = outputs.detach()
        labels = torch.as_tensor(true_labels, device=preds.device, dtype=torch.long)
        pred_classes = torch.argmax(preds, dim=1)
        masked = preds.clone()
        masked[torch.arange(preds.shape[0], device=preds.device), labels] = -torch.inf
        losses = preds[torch.arange(preds.shape[0], device=preds.device), labels] - torch.max(masked, dim=1).values
        return (
            (pred_classes != labels).cpu().numpy().astype(bool),
            losses.cpu().numpy().astype(np.float64),
            pred_classes.cpu().numpy().astype(np.int64),
        )

    preds_np = np.asarray(outputs)
    if preds_np.ndim == 1:
        preds_np = preds_np[None, :]
    pred_classes = np.argmax(preds_np, axis=1)
    masked = preds_np.copy()
    masked[np.arange(preds_np.shape[0]), true_labels] = -np.inf
    losses = preds_np[np.arange(preds_np.shape[0]), true_labels] - np.max(masked, axis=1)
    return pred_classes != true_labels, losses.astype(np.float64), pred_classes.astype(np.int64)


def accept_patch_updates(
    current_adversarial: np.ndarray,
    current_loss: np.ndarray,
    current_l2: np.ndarray,
    candidate_adversarial: np.ndarray,
    candidate_loss: np.ndarray,
    candidate_l2: np.ndarray,
) -> np.ndarray:
    both_adversarial = current_adversarial & candidate_adversarial
    return np.where(both_adversarial, candidate_l2 < current_l2, candidate_loss < current_loss)


def accept_location_updates(
    current_adversarial: np.ndarray,
    current_loss: np.ndarray,
    current_l2: np.ndarray,
    candidate_adversarial: np.ndarray,
    candidate_loss: np.ndarray,
    candidate_l2: np.ndarray,
    query: int,
    temp: float,
) -> np.ndarray:
    accepted = np.zeros_like(current_adversarial, dtype=bool)
    both_adversarial = current_adversarial & candidate_adversarial
    accepted[both_adversarial] = candidate_l2[both_adversarial] < current_l2[both_adversarial]

    rest = ~both_adversarial
    improved = rest & (candidate_loss < current_loss)
    accepted[improved] = True

    curr_temp = temp / (query + 1)
    if curr_temp > 0:
        metropolis_mask = rest & ~improved
        diff = candidate_loss[metropolis_mask] - current_loss[metropolis_mask]
        probs = np.exp(-diff / curr_temp)
        accepted[metropolis_mask] = np.random.rand(probs.shape[0]) < probs
    return accepted


def save_result(
    save_prefix: Path,
    x_orig: np.ndarray,
    x_adv: np.ndarray,
    adversarial: bool,
    queries: int,
    loc: np.ndarray,
    initial_loc: Optional[np.ndarray],
    patch: np.ndarray,
    patch_geno: np.ndarray,
    true_label: int,
    final_prediction: int,
    first_success_query: Optional[int],
    eps_linf: Optional[float],
    fixed_location: bool,
    location_source: Optional[str],
    attack_model: str,
    attack_model_source: str,
    attack_model_index: object,
    attack_model_name: str,
    patch_size: int,
    position_rule: str,
    save_images: bool,
) -> None:
    save_prefix.parent.mkdir(parents=True, exist_ok=True)
    orig_patch = x_orig[loc[0]: loc[0] + patch.shape[0], loc[1]: loc[1] + patch.shape[1], :]
    data = {
        "orig": x_orig,
        "adversary": x_adv,
        "adversarial": bool(adversarial),
        "queries": int(queries),
        "loc": loc.copy(),
        "patch": patch.copy(),
        "patch_width": int(patch.shape[0]),
        "eps_linf": eps_linf,
        "attack_model": attack_model,
        "attack_model_source": attack_model_source,
        "attack_model_index": attack_model_index,
        "attack_model_name": attack_model_name,
        "model": attack_model_name,
        "patch_size": int(patch_size),
        "position_rule": position_rule,
        "fixed_location": bool(fixed_location),
        "location_source": location_source,
        "initial_loc": None if initial_loc is None else initial_loc.copy(),
        "true_label": int(true_label),
        "first_success_query": None if first_success_query is None else int(first_success_query),
        "patch_position_y": int(loc[0]),
        "patch_position_x": int(loc[1]),
        "patch_position_h": int(patch.shape[0]),
        "patch_position_w": int(patch.shape[1]),
        "final_l2": l2(patch, orig_patch),
        "final_linf": linf(patch, orig_patch),
        "final_prediction": int(final_prediction),
        "patch_genotype": patch_geno.copy(),
        "process": [],
    }
    np.save(str(save_prefix) + ".npy", data, allow_pickle=True)
    if save_images:
        save_rgb_image(save_prefix.with_name(save_prefix.name + "_adversary.png"), x_adv)
        save_rgb_image(save_prefix.with_name(save_prefix.name + "_patch.png"), patch)


def run_strict_one_plus_one_batch(
    model: ImageNetModel,
    x_batch: np.ndarray,
    true_labels: np.ndarray,
    save_prefixes: List[Path],
    initial_locs: Optional[np.ndarray],
    args: argparse.Namespace,
    attack_model_name: str,
    attack_model_source: str,
) -> List[Dict[str, object]]:
    batch_size, h, w, _ = x_batch.shape
    patch_size = int(args.s)
    patch_genos = np.random.rand(batch_size, args.N, 7)
    if initial_locs is None:
        locs = np.random.randint(h - patch_size, size=(batch_size, 2))
        saved_initial_locs: List[Optional[np.ndarray]] = [None for _ in range(batch_size)]
    else:
        locs = np.clip(initial_locs.astype(np.int64), 0, h - patch_size)
        saved_initial_locs = [loc.copy() for loc in locs]

    patches = render_patches_for_batch(x_batch, patch_genos, locs, patch_size, args.linf)
    x_adv = apply_patches_to_batch(x_batch, patches, locs)
    current_adv, current_loss, current_pred = evaluate_batch(model, x_adv, true_labels)
    current_l2 = patch_l2_batch(x_batch, patches, locs)
    first_success_query = np.where(current_adv, 1, 0).astype(np.int64)

    patch_counter = 0
    for query in tqdm(range(1, args.queries), desc="strict (1+1)-ES"):
        previous_adv = current_adv.copy()
        patch_counter += 1
        if args.fixed_position or patch_counter < args.li:
            candidate_genos = np.stack([mutate(patch_genos[idx], args.mut) for idx in range(batch_size)], axis=0)
            candidate_locs = locs
            candidate_patches = render_patches_for_batch(x_batch, candidate_genos, candidate_locs, patch_size, args.linf)
            candidate_x_adv = apply_patches_to_batch(x_batch, candidate_patches, candidate_locs)
            candidate_adv, candidate_loss, candidate_pred = evaluate_batch(model, candidate_x_adv, true_labels)
            candidate_l2 = patch_l2_batch(x_batch, candidate_patches, candidate_locs)
            accepted = accept_patch_updates(
                current_adv,
                current_loss,
                current_l2,
                candidate_adv,
                candidate_loss,
                candidate_l2,
            )
            patch_genos[accepted] = candidate_genos[accepted]
        else:
            patch_counter = 0
            sh_i = int(max(sh_selection(args.queries, query) * h, 0))
            candidate_locs = np.stack(
                [update_location(locs[idx].copy(), sh_i, h, patch_size) for idx in range(batch_size)],
                axis=0,
            )
            candidate_patches = render_patches_for_batch(x_batch, patch_genos, candidate_locs, patch_size, args.linf)
            candidate_x_adv = apply_patches_to_batch(x_batch, candidate_patches, candidate_locs)
            candidate_adv, candidate_loss, candidate_pred = evaluate_batch(model, candidate_x_adv, true_labels)
            candidate_l2 = patch_l2_batch(x_batch, candidate_patches, candidate_locs)
            accepted = accept_location_updates(
                current_adv,
                current_loss,
                current_l2,
                candidate_adv,
                candidate_loss,
                candidate_l2,
                query,
                args.temp,
            )
            locs[accepted] = candidate_locs[accepted]

        patches[accepted] = candidate_patches[accepted]
        x_adv[accepted] = candidate_x_adv[accepted]
        current_adv[accepted] = candidate_adv[accepted]
        current_loss[accepted] = candidate_loss[accepted]
        current_pred[accepted] = candidate_pred[accepted]
        current_l2[accepted] = candidate_l2[accepted]
        new_success = (~previous_adv) & current_adv & (first_success_query == 0)
        first_success_query[new_success] = query + 1

    final_adv, _, final_pred = evaluate_batch(model, x_adv, true_labels)
    final_new_success = final_adv & (first_success_query == 0)
    first_success_query[final_new_success] = args.queries
    current_adv = final_adv
    final_linf = patch_linf_batch(x_batch, patches, locs)
    rows: List[Dict[str, object]] = []
    for idx in range(batch_size):
        save_result(
            save_prefixes[idx],
            x_orig=x_batch[idx],
            x_adv=x_adv[idx],
            adversarial=bool(current_adv[idx]),
            queries=args.queries,
            loc=locs[idx],
            initial_loc=saved_initial_locs[idx],
            patch=patches[idx],
            patch_geno=patch_genos[idx],
            true_label=int(true_labels[idx]),
            final_prediction=int(final_pred[idx]),
            first_success_query=(
                int(first_success_query[idx])
                if int(first_success_query[idx]) > 0 and bool(current_adv[idx])
                else None
            ),
            eps_linf=args.linf,
            fixed_location=args.fixed_position,
            location_source=args.location_source,
            attack_model=attack_model_name,
            attack_model_source=attack_model_source,
            attack_model_index=args.model,
            attack_model_name=getattr(model, "model_name", str(args.model)),
            patch_size=patch_size,
            position_rule=args.position_rule,
            save_images=args.save_images,
        )
        rows.append(
            {
                "index": args.current_indices[idx],
                "image_path": args.current_image_paths[idx],
                "output_prefix": str(save_prefixes[idx]),
                "attack_model": attack_model_name,
                "model": getattr(model, "model_name", str(args.model)),
                "model_source": attack_model_source,
                "patch_size": int(patch_size),
                "position_rule": args.position_rule,
                "true_label": int(true_labels[idx]),
                "adversarial": int(bool(current_adv[idx])),
                "first_success_query": (
                    int(first_success_query[idx])
                    if int(first_success_query[idx]) > 0 and bool(current_adv[idx])
                    else ""
                ),
                "final_prediction": int(final_pred[idx]),
                "queries": int(args.queries),
                "loc_y": int(locs[idx, 0]),
                "loc_x": int(locs[idx, 1]),
                "patch_position_y": int(locs[idx, 0]),
                "patch_position_x": int(locs[idx, 1]),
                "patch_position_h": int(patch_size),
                "patch_position_w": int(patch_size),
                "initial_loc_y": "" if saved_initial_locs[idx] is None else int(saved_initial_locs[idx][0]),
                "initial_loc_x": "" if saved_initial_locs[idx] is None else int(saved_initial_locs[idx][1]),
                "fixed_location": int(args.fixed_position),
                "location_source": args.location_source or "",
                "eps_linf": "" if args.linf is None else float(args.linf),
                "final_l2": float(current_l2[idx]),
                "final_linf": float(final_linf[idx]),
            }
        )
    return rows


def write_csv_rows(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(summary_path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    write_csv_rows(summary_path, rows, list(rows[0].keys()))


def _success_query(row: Dict[str, object]) -> Optional[int]:
    value = row.get("first_success_query", "")
    if value in ("", None):
        return None
    try:
        query = int(float(value))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(query) or query <= 0:
        return None
    return query


def write_success_tables(save_root: Path, rows: List[Dict[str, object]]) -> None:
    event_fieldnames = [
        "first_success_query",
        "index",
        "image_path",
        "output_prefix",
        "attack_model",
        "model",
        "model_source",
        "patch_size",
        "position_rule",
        "true_label",
        "final_prediction",
        "queries",
        "loc_y",
        "loc_x",
        "patch_position_y",
        "patch_position_x",
        "patch_position_h",
        "patch_position_w",
        "fixed_location",
        "location_source",
        "eps_linf",
    ]
    events: List[Dict[str, object]] = []
    for row in rows:
        if int(row.get("adversarial", 0) or 0) != 1:
            continue
        query = _success_query(row)
        if query is None:
            continue
        event = {name: row.get(name, "") for name in event_fieldnames}
        event["first_success_query"] = query
        events.append(event)
    events.sort(key=lambda item: (int(item["first_success_query"]), int(item["index"])))
    write_csv_rows(save_root / "success_events.csv", events, event_fieldnames)

    grouped: Dict[int, List[Dict[str, object]]] = {}
    for event in events:
        grouped.setdefault(int(event["first_success_query"]), []).append(event)

    by_query_rows: List[Dict[str, object]] = []
    cumulative = 0
    for query in sorted(grouped):
        query_events = grouped[query]
        cumulative += len(query_events)
        by_query_rows.append(
            {
                "first_success_query": query,
                "new_successes": len(query_events),
                "cumulative_successes": cumulative,
                "image_indices": ";".join(str(event["index"]) for event in query_events),
                "image_paths": ";".join(str(event["image_path"]) for event in query_events),
            }
        )
    write_csv_rows(
        save_root / "success_by_query.csv",
        by_query_rows,
        [
            "first_success_query",
            "new_successes",
            "cumulative_successes",
            "image_indices",
            "image_paths",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run strict CamoPatch (1+1)-ES on multiple CSV images per model forward.",
    )
    parser.add_argument("--images_csv", "--images-csv", dest="images_csv", required=True)
    parser.add_argument("--save_root", "--save-root", dest="save_root", required=True)
    parser.add_argument(
        "--model",
        default="1",
        help="B-cos/torchvision model name. Legacy indices still work: 1 is ResNet-50.",
    )
    parser.add_argument("--model_source", choices=("auto", "bcos", "torchvision"), default="bcos")
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--temp", type=float, default=300.0)
    parser.add_argument("--mut", type=float, default=0.3)
    parser.add_argument("--s", type=int, default=16)
    parser.add_argument("--queries", type=int, default=10000)
    parser.add_argument("--li", type=int, default=4)
    parser.add_argument("--linf", type=parse_linf, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--image_batch_size", "--image-batch-size", dest="image_batch_size", type=int, default=8)
    parser.add_argument(
        "--limit_images",
        "--limit-images",
        dest="limit_images",
        type=int,
        default=0,
        help="Optional smoke-test limit. 0 means use every image in --images_csv.",
    )
    parser.add_argument(
        "--position_rule",
        "--position-rule",
        dest="position_rule",
        choices=POSITION_RULES,
        default=None,
        help=(
            "Initial patch location rule. Default is random unless --init_bcos_position "
            "or --fixed_bcos_position is used. Non-random rules use B-cos explanations."
        ),
    )
    parser.add_argument(
        "--fixed_position",
        "--fixed-position",
        dest="fixed_position",
        action="store_true",
        help="Keep the initial patch location fixed for all queries.",
    )
    parser.add_argument("--init_bcos_position", action="store_true")
    parser.add_argument("--fixed_bcos_position", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_save_images", action="store_false", dest="save_images")
    parser.set_defaults(save_images=True)
    args = parser.parse_args()

    if args.image_batch_size <= 0:
        parser.error("--image_batch_size must be > 0")
    if args.limit_images < 0:
        parser.error("--limit_images must be >= 0")
    if args.fixed_bcos_position and args.position_rule not in (None, "margin"):
        parser.error("--fixed_bcos_position is an alias for --position-rule margin --fixed-position.")
    if args.init_bcos_position and args.position_rule not in (None, "margin"):
        parser.error("--init_bcos_position is an alias for --position-rule margin.")
    if args.position_rule is None:
        args.position_rule = "margin" if (args.init_bcos_position or args.fixed_bcos_position) else "random"
    if args.fixed_bcos_position:
        args.fixed_position = True
    if args.seed is not None:
        np.random.seed(args.seed)

    load_image = transforms.Compose([
        transforms.Resize(IMAGENET_SL),
        transforms.CenterCrop(IMAGENET_SL),
        transforms.ToTensor(),
    ])

    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    items = load_items_from_csv(Path(args.images_csv))
    if args.limit_images:
        items = items[:args.limit_images]
        if not items:
            parser.error("--limit_images removed every image from the run.")

    model = ImageNetModel(args.model, device=args.device, model_source=args.model_source)
    attack_model_source = model.model_source
    attack_model_name = describe_attack_model(args.model, attack_model_source)
    print(f"Attacking model: {attack_model_name}")
    print("Algorithm: strict per-image (1+1)-ES; batch dimension is images, not candidates.")
    print(f"Position rule: {args.position_rule}; fixed_position={args.fixed_position}")

    bcos_guide = None
    if args.position_rule != "random":
        bcos_guide = resolve_bcos_guide_model(model, args.model, args.device)

    all_rows: List[Dict[str, object]] = []
    for start in range(0, len(items), args.image_batch_size):
        chunk = items[start:start + args.image_batch_size]
        indices = [item[0] for item in chunk]
        true_labels = np.array([item[1] for item in chunk], dtype=np.int64)
        stems = [item[2] for item in chunk]
        image_paths = [item[3] for item in chunk]
        save_prefixes = [
            save_root / f"{indices[idx]:05d}_{stems[idx]}_label_{true_labels[idx]}_strict_1p1"
            for idx in range(len(chunk))
        ]

        print(f"\nBatch images {start + 1}-{start + len(chunk)}/{len(items)}")
        x_batch, x_chw_tensors = load_image_batch(image_paths, load_image)

        initial_locs = None
        args.location_source = None if args.position_rule == "random" else bcos_location_source(
            args.position_rule,
            args.fixed_position,
        )
        if args.position_rule == "random" and args.fixed_position:
            _, h, _, _ = x_batch.shape
            if h <= args.s:
                initial_locs = np.zeros((len(chunk), 2), dtype=np.int64)
            else:
                initial_locs = np.random.randint(h - args.s, size=(len(chunk), 2))
            args.location_source = "random_fixed"
            for idx, loc in enumerate(initial_locs):
                print(f"  image {indices[idx]:05d}: random fixed init loc=({int(loc[0])},{int(loc[1])})")
        elif args.position_rule != "random":
            initial_locs_list = []
            for idx, x_chw in enumerate(x_chw_tensors):
                loc, guide_prediction, secondary_class = resolve_bcos_position_by_rule(
                    model,
                    x_chw,
                    int(true_labels[idx]),
                    args.s,
                    args.model,
                    args.device,
                    args.position_rule,
                    guide=bcos_guide,
                )
                print(
                    f"  image {indices[idx]:05d}: {args.position_rule} init loc=({int(loc[0])},{int(loc[1])}) "
                    f"guide_pred={guide_prediction} secondary={secondary_class}"
                )
                initial_locs_list.append(loc)
            initial_locs = np.stack(initial_locs_list, axis=0)

        args.current_indices = indices
        args.current_image_paths = image_paths
        rows = run_strict_one_plus_one_batch(
            model,
            x_batch,
            true_labels,
            save_prefixes,
            initial_locs,
            args,
            attack_model_name,
            attack_model_source,
        )
        all_rows.extend(rows)
        write_summary(save_root / "summary.csv", all_rows)
        write_success_tables(save_root, all_rows)
        successes = sum(int(row["adversarial"]) for row in all_rows)
        print(f"Summary so far: {successes}/{len(all_rows)} successful attacks")

    print(f"\nDone. Summary: {save_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
