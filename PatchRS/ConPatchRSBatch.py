"""Patch-only Sparse-RS adapter for the B-cos Kaggle matrix.

The search loop mirrors the image-specific `norm='patches'` branch from
fra31/sparse-rs, with repo-specific additions for B-cos/GradCAM initial
locations, fixed versus movable locations, optional L_inf projection, and the
CamoPatch result zip contract.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMOPATCH_DIR = REPO_ROOT / "CamoPatch"
for _path in (REPO_ROOT, CAMOPATCH_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from CamoPatch import l2, linf, save_rgb_image  # noqa: E402
from ConCamoPatch import describe_attack_model, parse_linf  # noqa: E402
from ConCamoPatchBatch import (  # noqa: E402
    bcos_location_source,
    load_items_from_csv,
    resolve_bcos_position_by_rule,
    write_success_tables,
    write_summary,
)
from ImageNetModels import ImageNetModel  # noqa: E402


IMAGENET_SL = 224
POSITION_RULES = ("random", "margin", "top1", "dynamic-margin", "dynamic", "gradcam")
PATCH_INITS = ("random_squares", "random", "uniform", "stripes", "sh")


def as_nhwc(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().permute(1, 2, 0).numpy()


def random_choice(shape: Tuple[int, ...] | List[int], device: torch.device) -> torch.Tensor:
    return torch.sign(2 * torch.rand(shape, device=device) - 1).clamp(0.0, 1.0)


def p_selection(
    it: int,
    n_queries: int,
    p_init: float,
    resc_schedule: bool,
    constant_schedule: bool,
) -> float:
    if resc_schedule:
        it = int(it / n_queries * 10000)

    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it:
        p = p_init / 512
    else:
        p = p_init

    return p_init if constant_schedule else p


def sh_selection(n_queries: int, it: int) -> float:
    return max((float(n_queries - it) / n_queries) ** 1.0, 0.0) * 0.75


def get_init_patches(
    batch_size: int,
    channels: int,
    patch_size: int,
    init_mode: str,
    init_iters: int,
    device: torch.device,
) -> torch.Tensor:
    if init_mode == "stripes":
        return random_choice([batch_size, channels, 1, patch_size], device).expand(
            batch_size, channels, patch_size, patch_size
        ).clone()
    if init_mode == "uniform":
        return random_choice([batch_size, channels, 1, 1], device).expand(
            batch_size, channels, patch_size, patch_size
        ).clone()
    if init_mode == "random":
        return random_choice([batch_size, channels, patch_size, patch_size], device)
    if init_mode == "sh":
        return torch.ones([batch_size, channels, patch_size, patch_size], device=device)
    if init_mode != "random_squares":
        raise ValueError(f"Unsupported patch init: {init_mode}")

    patches = torch.zeros([batch_size, channels, patch_size, patch_size], device=device)
    max_square = max(2, int(math.ceil(patch_size ** 0.5)))
    for image_idx in range(batch_size):
        for _ in range(init_iters):
            size_init = int(torch.randint(low=1, high=max_square, size=[1], device=device).item())
            loc_y = int(torch.randint(patch_size - size_init + 1, size=[1], device=device).item())
            loc_x = int(torch.randint(patch_size - size_init + 1, size=[1], device=device).item())
            patches[
                image_idx,
                :,
                loc_y: loc_y + size_init,
                loc_x: loc_x + size_init,
            ] = random_choice([channels, 1, 1], device)
    return patches.clamp(0.0, 1.0)


def project_patches_linf(
    x_orig: torch.Tensor,
    patches: torch.Tensor,
    locs: torch.Tensor,
    eps_linf: Optional[float],
) -> torch.Tensor:
    patches = patches.clamp(0.0, 1.0)
    if eps_linf is None:
        return patches

    projected = patches.clone()
    patch_size = patches.shape[-1]
    eps = float(eps_linf)
    for idx in range(patches.shape[0]):
        y = int(locs[idx, 0].item())
        x = int(locs[idx, 1].item())
        orig_patch = x_orig[idx, :, y: y + patch_size, x: x + patch_size]
        projected[idx] = torch.minimum(
            torch.maximum(projected[idx], orig_patch - eps),
            orig_patch + eps,
        ).clamp(0.0, 1.0)
    return projected


def apply_patches(
    x_orig: torch.Tensor,
    patches: torch.Tensor,
    locs: torch.Tensor,
    eps_linf: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    patches = project_patches_linf(x_orig, patches, locs, eps_linf)
    x_adv = x_orig.clone()
    patch_size = patches.shape[-1]
    for idx in range(x_orig.shape[0]):
        y = int(locs[idx, 0].item())
        x = int(locs[idx, 1].item())
        x_adv[idx, :, y: y + patch_size, x: x + patch_size] = patches[idx]
    return x_adv.clamp(0.0, 1.0), patches


def patch_l2_linf_batch(
    x_orig: torch.Tensor,
    patches: torch.Tensor,
    locs: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    patch_size = patches.shape[-1]
    l2_values = np.empty((patches.shape[0],), dtype=np.float64)
    linf_values = np.empty((patches.shape[0],), dtype=np.float64)
    for idx in range(patches.shape[0]):
        y = int(locs[idx, 0].item())
        x = int(locs[idx, 1].item())
        orig_patch = as_nhwc(x_orig[idx, :, y: y + patch_size, x: x + patch_size])
        patch = as_nhwc(patches[idx])
        l2_values[idx] = l2(patch, orig_patch)
        linf_values[idx] = linf(patch, orig_patch)
    return l2_values, linf_values


def evaluate_batch(
    model: ImageNetModel,
    x_batch: torch.Tensor,
    true_labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = model.predict(x_batch)
    labels = true_labels.to(device=outputs.device, dtype=torch.long)
    pred_classes = outputs.argmax(dim=1)
    masked = outputs.clone()
    masked[torch.arange(outputs.shape[0], device=outputs.device), labels] = -torch.inf
    margins = outputs[torch.arange(outputs.shape[0], device=outputs.device), labels] - masked.max(dim=1).values
    return pred_classes != labels, margins, pred_classes


def build_initial_locs(
    x_batch: torch.Tensor,
    initial_locs: Optional[np.ndarray],
    patch_size: int,
) -> Tuple[torch.Tensor, List[Optional[np.ndarray]]]:
    batch_size, _, h, _ = x_batch.shape
    device = x_batch.device
    if initial_locs is not None:
        locs = torch.as_tensor(initial_locs, device=device, dtype=torch.long)
        locs = locs.clamp(0, h - patch_size)
        saved = [loc.detach().cpu().numpy().astype(np.int64).copy() for loc in locs]
        return locs, saved

    if h <= patch_size:
        locs = torch.zeros([batch_size, 2], device=device, dtype=torch.long)
    else:
        locs = torch.randint(h - patch_size, size=[batch_size, 2], device=device)
    return locs, [None for _ in range(batch_size)]


def save_result(
    save_prefix: Path,
    x_orig: torch.Tensor,
    x_adv: torch.Tensor,
    adversarial: bool,
    queries: int,
    loc: torch.Tensor,
    initial_loc: Optional[np.ndarray],
    patch: torch.Tensor,
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
    patch_init: str,
    save_images: bool,
) -> None:
    save_prefix.parent.mkdir(parents=True, exist_ok=True)
    loc_np = loc.detach().cpu().numpy().astype(np.int64)
    x_orig_np = as_nhwc(x_orig)
    x_adv_np = as_nhwc(x_adv)
    patch_np = as_nhwc(patch)
    orig_patch = x_orig_np[
        loc_np[0]: loc_np[0] + patch_np.shape[0],
        loc_np[1]: loc_np[1] + patch_np.shape[1],
        :,
    ]
    data = {
        "orig": x_orig_np,
        "adversary": x_adv_np,
        "adversarial": bool(adversarial),
        "queries": int(queries),
        "loc": loc_np.copy(),
        "patch": patch_np.copy(),
        "patch_width": int(patch_np.shape[0]),
        "eps_linf": eps_linf,
        "attack": "patchrs",
        "attack_model": attack_model,
        "attack_model_source": attack_model_source,
        "attack_model_index": attack_model_index,
        "attack_model_name": attack_model_name,
        "model": attack_model_name,
        "patch_size": int(patch_size),
        "position_rule": position_rule,
        "patch_init": patch_init,
        "fixed_location": bool(fixed_location),
        "location_source": location_source,
        "initial_loc": None if initial_loc is None else initial_loc.copy(),
        "true_label": int(true_label),
        "first_success_query": None if first_success_query is None else int(first_success_query),
        "patch_position_y": int(loc_np[0]),
        "patch_position_x": int(loc_np[1]),
        "patch_position_h": int(patch_np.shape[0]),
        "patch_position_w": int(patch_np.shape[1]),
        "final_l2": l2(patch_np, orig_patch),
        "final_linf": linf(patch_np, orig_patch),
        "final_prediction": int(final_prediction),
        "process": [],
    }
    np.save(str(save_prefix) + ".npy", data, allow_pickle=True)
    if save_images:
        save_rgb_image(save_prefix.with_name(save_prefix.name + "_adversary.png"), x_adv_np)
        save_rgb_image(save_prefix.with_name(save_prefix.name + "_patch.png"), patch_np)


def run_patch_rs_batch(
    model: ImageNetModel,
    x_batch: torch.Tensor,
    true_labels: torch.Tensor,
    save_prefixes: List[Path],
    initial_locs: Optional[np.ndarray],
    args: argparse.Namespace,
    attack_model_name: str,
    attack_model_source: str,
) -> List[Dict[str, object]]:
    batch_size, channels, h, _ = x_batch.shape
    patch_size = int(args.s)
    device = x_batch.device
    locs, saved_initial_locs = build_initial_locs(x_batch, initial_locs, patch_size)
    patches = get_init_patches(
        batch_size=batch_size,
        channels=channels,
        patch_size=patch_size,
        init_mode=args.patch_init,
        init_iters=args.patch_init_iters,
        device=device,
    )
    x_best, patches = apply_patches(x_batch, patches, locs, args.linf)
    current_success, margin_min, current_pred = evaluate_batch(model, x_best, true_labels)
    loss_min = margin_min.clone()
    first_success_query = torch.where(
        current_success,
        torch.ones_like(true_labels, device=device, dtype=torch.long),
        torch.zeros_like(true_labels, device=device, dtype=torch.long),
    )
    n_queries = torch.ones_like(true_labels, device=device, dtype=torch.long)

    it_start_cu = 0
    for query in range(args.queries):
        s_it = int(max(p_selection(query, args.queries, args.alpha_init, args.rescale_schedule, args.constant_schedule) ** 0.5 * patch_size, 1))
        it_start_cu = query
        if s_it == 1:
            break
    it_start_cu = it_start_cu + (args.queries - it_start_cu) // 2
    loc_period = abs(int(args.li))

    for query in tqdm(range(1, args.queries), desc="Patch-RS patches"):
        idx_to_fool = (~current_success).nonzero(as_tuple=False).flatten()
        if idx_to_fool.numel() == 0:
            break

        x_curr = x_batch[idx_to_fool]
        patches_curr = patches[idx_to_fool]
        loc_curr = locs[idx_to_fool]
        labels_curr = true_labels[idx_to_fool]
        loss_min_curr = loss_min[idx_to_fool]

        s_it = int(max(p_selection(query, args.queries, args.alpha_init, args.rescale_schedule, args.constant_schedule) ** 0.5 * patch_size, 1))
        p_y = int(torch.randint(patch_size - s_it + 1, size=[1], device=device).item())
        p_x = int(torch.randint(patch_size - s_it + 1, size=[1], device=device).item())
        sh_it = int(max(sh_selection(args.queries, query) * h, 0))

        update_loc = (query % loc_period == 0) and (sh_it > 0) and not args.fixed_position
        if args.inverse_location_schedule and sh_it > 0 and not args.fixed_position:
            update_loc = not update_loc
        update_patch = not update_loc

        patches_new = patches_curr.clone()
        loc_new = loc_curr.clone()
        if update_patch:
            for counter in range(patches_new.shape[0]):
                if query < it_start_cu and s_it > 1:
                    patches_new[
                        counter,
                        :,
                        p_y: p_y + s_it,
                        p_x: p_x + s_it,
                    ] += random_choice([channels, 1, 1], device)
                else:
                    old_color = patches_new[counter, :, p_y: p_y + s_it, p_x: p_x + s_it].clone()
                    if query < it_start_cu:
                        new_color = old_color.clone()
                        while bool((new_color == old_color).all().item()):
                            new_color = random_choice([channels, 1, 1], device)
                        patches_new[counter, :, p_y: p_y + s_it, p_x: p_x + s_it] = new_color
                    else:
                        new_channel = int(torch.randint(low=0, high=channels, size=[1], device=device).item())
                        patches_new[
                            counter,
                            new_channel,
                            p_y: p_y + s_it,
                            p_x: p_x + s_it,
                        ] = 1.0 - patches_new[
                            counter,
                            new_channel,
                            p_y: p_y + s_it,
                            p_x: p_x + s_it,
                        ]
            patches_new.clamp_(0.0, 1.0)
        else:
            shifts = torch.randint(low=-sh_it, high=sh_it + 1, size=loc_new.shape, device=device)
            loc_new = (loc_new + shifts).clamp(0, h - patch_size)

        x_new, patches_new = apply_patches(x_curr, patches_new, loc_new, args.linf)
        candidate_success, margin, candidate_pred = evaluate_batch(model, x_new, labels_curr)
        n_queries[idx_to_fool] += 1

        improved_loss = loss_min_curr > margin
        misclassified = margin < -1e-6
        accepted = improved_loss | misclassified
        if accepted.any():
            accepted_global = idx_to_fool[accepted]
            loss_update_global = idx_to_fool[improved_loss]
            loss_min[loss_update_global] = margin[improved_loss]
            margin_min[accepted_global] = margin[accepted]
            patches[accepted_global] = patches_new[accepted]
            locs[accepted_global] = loc_new[accepted]
            x_best[accepted_global] = x_new[accepted]
            current_pred[accepted_global] = candidate_pred[accepted]
            current_success[accepted_global] = candidate_success[accepted]

        new_success = accepted & candidate_success & (first_success_query[idx_to_fool] == 0)
        if new_success.any():
            first_success_query[idx_to_fool[new_success]] = query + 1

    final_success, _, final_pred = evaluate_batch(model, x_best, true_labels)
    final_new_success = final_success & (first_success_query == 0)
    first_success_query[final_new_success] = n_queries[final_new_success]
    current_success = final_success
    current_pred = final_pred
    final_l2, final_linf = patch_l2_linf_batch(x_batch, patches, locs)

    rows: List[Dict[str, object]] = []
    for idx in range(batch_size):
        success_query = (
            int(first_success_query[idx].item())
            if int(first_success_query[idx].item()) > 0 and bool(current_success[idx].item())
            else None
        )
        save_result(
            save_prefixes[idx],
            x_orig=x_batch[idx],
            x_adv=x_best[idx],
            adversarial=bool(current_success[idx].item()),
            queries=args.queries,
            loc=locs[idx],
            initial_loc=saved_initial_locs[idx],
            patch=patches[idx],
            true_label=int(true_labels[idx].item()),
            final_prediction=int(current_pred[idx].item()),
            first_success_query=success_query,
            eps_linf=args.linf,
            fixed_location=args.fixed_position,
            location_source=args.location_source,
            attack_model=attack_model_name,
            attack_model_source=attack_model_source,
            attack_model_index=args.model,
            attack_model_name=getattr(model, "model_name", str(args.model)),
            patch_size=patch_size,
            position_rule=args.position_rule,
            patch_init=args.patch_init,
            save_images=args.save_images,
        )
        rows.append(
            {
                "index": args.current_indices[idx],
                "image_path": args.current_image_paths[idx],
                "output_prefix": str(save_prefixes[idx]),
                "attack": "patchrs",
                "attack_model": attack_model_name,
                "model": getattr(model, "model_name", str(args.model)),
                "model_source": attack_model_source,
                "patch_size": int(patch_size),
                "position_rule": args.position_rule,
                "patch_init": args.patch_init,
                "true_label": int(true_labels[idx].item()),
                "adversarial": int(bool(current_success[idx].item())),
                "first_success_query": "" if success_query is None else success_query,
                "final_prediction": int(current_pred[idx].item()),
                "queries": int(args.queries),
                "loc_y": int(locs[idx, 0].item()),
                "loc_x": int(locs[idx, 1].item()),
                "patch_position_y": int(locs[idx, 0].item()),
                "patch_position_x": int(locs[idx, 1].item()),
                "patch_position_h": int(patch_size),
                "patch_position_w": int(patch_size),
                "initial_loc_y": "" if saved_initial_locs[idx] is None else int(saved_initial_locs[idx][0]),
                "initial_loc_x": "" if saved_initial_locs[idx] is None else int(saved_initial_locs[idx][1]),
                "fixed_location": int(args.fixed_position),
                "location_source": args.location_source or "",
                "eps_linf": "" if args.linf is None else float(args.linf),
                "final_l2": float(final_l2[idx]),
                "final_linf": float(final_linf[idx]),
            }
        )
    return rows


def load_image_batch(
    image_paths: List[str],
    load_image: transforms.Compose,
    device: torch.device,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    chw_tensors: List[torch.Tensor] = []
    for image_path in image_paths:
        x_chw = load_image(Image.open(image_path).convert("RGB"))
        chw_tensors.append(x_chw)
    x_batch = torch.stack(chw_tensors, dim=0).to(device=device, dtype=torch.float32)
    return x_batch, chw_tensors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Sparse-RS patch-only attack on multiple CSV images per model forward.",
    )
    parser.add_argument("--images_csv", "--images-csv", dest="images_csv", required=True)
    parser.add_argument("--save_root", "--save-root", dest="save_root", required=True)
    parser.add_argument(
        "--model",
        default="1",
        help="B-cos/torchvision model name. Legacy indices still work: 1 is ResNet-50.",
    )
    parser.add_argument("--model_source", choices=("auto", "bcos", "torchvision"), default="bcos")
    parser.add_argument("--s", type=int, default=16)
    parser.add_argument("--queries", type=int, default=10000)
    parser.add_argument("--li", type=int, default=4, help="Location update period; ignored with --fixed-position.")
    parser.add_argument("--alpha_init", "--alpha-init", dest="alpha_init", type=float, default=0.4)
    parser.add_argument("--constant_schedule", "--constant-schedule", dest="constant_schedule", action="store_true")
    parser.add_argument("--no_rescale_schedule", "--no-rescale-schedule", dest="rescale_schedule", action="store_false")
    parser.set_defaults(rescale_schedule=True)
    parser.add_argument("--patch_init", "--patch-init", dest="patch_init", choices=PATCH_INITS, default="random_squares")
    parser.add_argument("--patch_init_iters", "--patch-init-iters", dest="patch_init_iters", type=int, default=1000)
    parser.add_argument("--linf", type=parse_linf, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--image_batch_size", "--image-batch-size", dest="image_batch_size", type=int, default=8)
    parser.add_argument("--limit_images", "--limit-images", dest="limit_images", type=int, default=0)
    parser.add_argument("--position_rule", "--position-rule", dest="position_rule", choices=POSITION_RULES, default="random")
    parser.add_argument("--fixed_position", "--fixed-position", dest="fixed_position", action="store_true")
    parser.add_argument("--inverse_location_schedule", "--inverse-location-schedule", dest="inverse_location_schedule", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_save_images", action="store_false", dest="save_images")
    parser.set_defaults(save_images=True)
    args = parser.parse_args()

    if args.image_batch_size <= 0:
        parser.error("--image_batch_size must be > 0")
    if args.limit_images < 0:
        parser.error("--limit_images must be >= 0")
    if args.s <= 0:
        parser.error("--s must be > 0")
    if args.patch_init_iters < 0:
        parser.error("--patch_init_iters must be >= 0")
    if not args.fixed_position and abs(int(args.li)) <= 1:
        parser.error("--li must have abs(value) > 1 when location movement is enabled")
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

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
    device = model.device
    attack_model_source = model.model_source
    attack_model_name = describe_attack_model(args.model, attack_model_source)
    print(f"Attacking model: {attack_model_name}")
    print("Algorithm: Sparse-RS patch-only; batch dimension is images.")
    print(
        f"Position rule: {args.position_rule}; fixed_position={args.fixed_position}; "
        f"patch_init={args.patch_init}"
    )

    bcos_guide = None
    if args.position_rule != "random":
        bcos_guide = model if getattr(model, "model_source", None) == "bcos" else ImageNetModel(args.model, device=args.device, model_source="bcos")

    all_rows: List[Dict[str, object]] = []
    for start in range(0, len(items), args.image_batch_size):
        chunk = items[start:start + args.image_batch_size]
        indices = [item[0] for item in chunk]
        true_labels = torch.tensor([item[1] for item in chunk], dtype=torch.long, device=device)
        stems = [item[2] for item in chunk]
        image_paths = [item[3] for item in chunk]
        save_prefixes = [
            save_root / f"{indices[idx]:05d}_{stems[idx]}_label_{int(true_labels[idx].item())}_patchrs"
            for idx in range(len(chunk))
        ]

        print(f"\nBatch images {start + 1}-{start + len(chunk)}/{len(items)}")
        x_batch, x_chw_tensors = load_image_batch(image_paths, load_image, device)

        initial_locs = None
        args.location_source = None if args.position_rule == "random" else bcos_location_source(
            args.position_rule,
            args.fixed_position,
        )
        if args.position_rule == "random" and args.fixed_position:
            _, _, h, _ = x_batch.shape
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
                    int(true_labels[idx].item()),
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
        rows = run_patch_rs_batch(
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
