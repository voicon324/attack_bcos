"""Adversarial Patch adapter for the B-cos Kaggle matrix.

This runner follows the targeted visible patch update from
A-LinCui/Adversarial_Patch_Attack, while keeping this repo's common Kaggle
contract: B-cos model loading, fixed/movable position rules, L_inf projection,
first-success-query bookkeeping, and zipped summary outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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
from LaVAN.ConLaVANBatch import (  # noqa: E402
    IMAGENET_SL,
    PATCH_INITS,
    POSITION_RULES,
    apply_patches,
    as_nhwc,
    build_initial_locs,
    init_patches,
    load_image_batch,
    model_logits,
    parse_step_size,
    patch_l2_linf_batch,
    project_patches_linf,
)


TARGET_MODES = ("fixed", "next", "random", "least_likely")
SUCCESS_MODES = ("untargeted", "targeted")


def evaluate_target_batch(
    model: ImageNetModel,
    x_batch: torch.Tensor,
    true_labels: torch.Tensor,
    target_labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        logits = model_logits(model, x_batch)
        true_labels = true_labels.to(device=logits.device, dtype=torch.long)
        target_labels = target_labels.to(device=logits.device, dtype=torch.long)
        pred = logits.argmax(dim=1)
        target_loss = F.cross_entropy(logits, target_labels, reduction="none")
        probs = torch.softmax(logits, dim=1)
        target_probability = probs.gather(1, target_labels[:, None]).squeeze(1)
        adversarial = pred != true_labels
        targeted = pred == target_labels
    return adversarial, target_loss.detach(), pred.detach(), targeted.detach(), target_probability.detach()


def build_target_labels(
    model: ImageNetModel,
    x_batch: torch.Tensor,
    true_labels: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    device = x_batch.device
    if args.target_mode == "fixed":
        target = torch.full_like(true_labels, int(args.target_class) % 1000)
    elif args.target_mode == "next":
        target = (true_labels + 1) % 1000
    elif args.target_mode == "random":
        target = torch.randint(0, 1000, size=true_labels.shape, device=device)
    elif args.target_mode == "least_likely":
        with torch.no_grad():
            target = model_logits(model, x_batch).argmin(dim=1).to(device=device)
    else:
        raise ValueError(f"Unsupported target mode: {args.target_mode}")

    equal = target == true_labels
    if equal.any():
        target = target.clone()
        target[equal] = (target[equal] + 1) % 1000
    return target.to(device=device, dtype=torch.long)


def success_mask(
    adversarial: torch.Tensor,
    targeted: torch.Tensor,
    target_probability: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.success_mode == "targeted":
        return targeted & (target_probability >= float(args.probability_threshold))
    return adversarial


def maybe_update_locations(
    model: ImageNetModel,
    x_orig: torch.Tensor,
    true_labels: torch.Tensor,
    target_labels: torch.Tensor,
    patches: torch.Tensor,
    locs: torch.Tensor,
    current_target_loss: torch.Tensor,
    current_success: torch.Tensor,
    query: int,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if args.fixed_position or query % abs(int(args.li)) != 0:
        return patches, locs, current_target_loss, current_success

    movable = (~current_success).nonzero(as_tuple=False).flatten()
    if movable.numel() == 0:
        return patches, locs, current_target_loss, current_success

    _, _, h, _ = x_orig.shape
    patch_size = patches.shape[-1]
    sh = int(max((float(args.queries - query) / max(args.queries, 1)) * 0.75 * h, 1))
    base_locs = locs[movable]
    shifts = torch.randint(low=-sh, high=sh + 1, size=base_locs.shape, device=locs.device)
    candidate_locs = (base_locs + shifts).clamp(0, h - patch_size)
    x_candidate, candidate_patches = apply_patches(
        x_orig[movable],
        patches[movable],
        candidate_locs,
        args.linf,
    )
    candidate_adv, candidate_loss, _, candidate_targeted, candidate_prob = evaluate_target_batch(
        model,
        x_candidate,
        true_labels[movable],
        target_labels[movable],
    )
    candidate_success = success_mask(candidate_adv, candidate_targeted, candidate_prob, args)
    accepted = (candidate_loss <= current_target_loss[movable]) | candidate_success
    if accepted.any():
        locs = locs.clone()
        patches = patches.clone()
        current_target_loss = current_target_loss.clone()
        current_success = current_success.clone()
        accepted_indices = movable[accepted]
        locs[accepted_indices] = candidate_locs[accepted]
        patches[accepted_indices] = candidate_patches[accepted]
        current_target_loss[accepted_indices] = candidate_loss[accepted]
        current_success[accepted_indices] = candidate_success[accepted]
    return patches, locs, current_target_loss, current_success


def save_result(
    save_prefix: Path,
    x_orig: torch.Tensor,
    x_adv: torch.Tensor,
    adversarial: bool,
    untargeted_success: bool,
    targeted_success: bool,
    queries: int,
    loc: torch.Tensor,
    initial_loc: np.ndarray,
    patch: torch.Tensor,
    true_label: int,
    target_class: int,
    final_prediction: int,
    first_success_query: Optional[int],
    target_probability: float,
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
    step_size: float,
    target_mode: str,
    success_mode: str,
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
        "untargeted_success": bool(untargeted_success),
        "targeted_success": bool(targeted_success),
        "queries": int(queries),
        "loc": loc_np.copy(),
        "patch": patch_np.copy(),
        "patch_width": int(patch_np.shape[0]),
        "eps_linf": eps_linf,
        "attack": "adversarial_patch",
        "attack_model": attack_model,
        "attack_model_source": attack_model_source,
        "attack_model_index": attack_model_index,
        "attack_model_name": attack_model_name,
        "model": attack_model_name,
        "patch_size": int(patch_size),
        "position_rule": position_rule,
        "patch_init": patch_init,
        "step_size": float(step_size),
        "target_mode": target_mode,
        "success_mode": success_mode,
        "fixed_location": bool(fixed_location),
        "location_source": location_source,
        "initial_loc": initial_loc.copy(),
        "true_label": int(true_label),
        "target_class": int(target_class),
        "first_success_query": None if first_success_query is None else int(first_success_query),
        "patch_position_y": int(loc_np[0]),
        "patch_position_x": int(loc_np[1]),
        "patch_position_h": int(patch_np.shape[0]),
        "patch_position_w": int(patch_np.shape[1]),
        "final_l2": l2(patch_np, orig_patch),
        "final_linf": linf(patch_np, orig_patch),
        "final_prediction": int(final_prediction),
        "target_probability": float(target_probability),
        "process": [],
    }
    np.save(str(save_prefix) + ".npy", data, allow_pickle=True)
    if save_images:
        save_rgb_image(save_prefix.with_name(save_prefix.name + "_adversary.png"), x_adv_np)
        save_rgb_image(save_prefix.with_name(save_prefix.name + "_patch.png"), patch_np)


def run_adversarial_patch_batch(
    model: ImageNetModel,
    x_batch: torch.Tensor,
    true_labels: torch.Tensor,
    save_prefixes: List[Path],
    initial_locs: Optional[np.ndarray],
    args: argparse.Namespace,
    attack_model_name: str,
    attack_model_source: str,
) -> List[Dict[str, object]]:
    batch_size, _, h, _ = x_batch.shape
    patch_size = int(args.s)
    device = x_batch.device
    locs, saved_initial_locs = build_initial_locs(batch_size, h, patch_size, device, initial_locs)
    target_labels = build_target_labels(model, x_batch, true_labels, args)
    patches = init_patches(x_batch, locs, patch_size, args.linf, args.patch_init)
    x_adv, patches = apply_patches(x_batch, patches, locs, args.linf)
    current_adv, current_target_loss, current_pred, current_targeted, current_prob = evaluate_target_batch(
        model,
        x_adv,
        true_labels,
        target_labels,
    )
    current_success = success_mask(current_adv, current_targeted, current_prob, args)
    first_success_query = torch.where(
        current_success,
        torch.ones_like(true_labels, device=device, dtype=torch.long),
        torch.zeros_like(true_labels, device=device, dtype=torch.long),
    )

    for query in tqdm(range(1, args.queries), desc="Adversarial patch"):
        active = (~current_success).nonzero(as_tuple=False).flatten()
        if active.numel() == 0:
            break

        x_active = x_batch[active].detach()
        loc_active = locs[active].detach()
        true_active = true_labels[active].detach()
        target_active = target_labels[active].detach()
        patch_var = patches[active].detach().clone().requires_grad_(True)
        x_candidate, _ = apply_patches(x_active, patch_var, loc_active, args.linf)
        logits = model_logits(model, x_candidate)
        loss = F.cross_entropy(logits, target_active, reduction="mean")
        loss.backward()
        if patch_var.grad is None:
            raise RuntimeError("Adversarial patch gradient is None")
        grad = patch_var.grad.detach()
        if args.gradient_mode == "raw":
            updated = patch_var.detach() - float(args.step_size) * grad
        else:
            updated = patch_var.detach() - float(args.step_size) * grad.sign()
        updated = project_patches_linf(x_active, updated, loc_active, args.linf)
        x_updated, updated = apply_patches(x_active, updated, loc_active, args.linf)
        cand_adv, cand_loss, cand_pred, cand_targeted, cand_prob = evaluate_target_batch(
            model,
            x_updated,
            true_active,
            target_active,
        )
        cand_success = success_mask(cand_adv, cand_targeted, cand_prob, args)

        patches[active] = updated.detach()
        x_adv[active] = x_updated.detach()
        current_target_loss[active] = cand_loss.detach()
        current_pred[active] = cand_pred.detach()
        current_adv[active] = cand_adv.detach()
        current_targeted[active] = cand_targeted.detach()
        current_prob[active] = cand_prob.detach()
        new_success = cand_success & (first_success_query[active] == 0)
        if new_success.any():
            first_success_query[active[new_success]] = query + 1
        current_success[active] = current_success[active] | cand_success.detach()

        patches, locs, current_target_loss, current_success = maybe_update_locations(
            model,
            x_batch,
            true_labels,
            target_labels,
            patches,
            locs,
            current_target_loss,
            current_success,
            query + 1,
            args,
        )
        move_new_success = current_success & (first_success_query == 0)
        if move_new_success.any():
            first_success_query[move_new_success] = query + 1
        x_adv, patches = apply_patches(x_batch, patches, locs, args.linf)

    final_adv, _, final_pred, final_targeted, final_prob = evaluate_target_batch(
        model,
        x_adv,
        true_labels,
        target_labels,
    )
    final_success = success_mask(final_adv, final_targeted, final_prob, args)
    final_new_success = final_success & (first_success_query == 0)
    first_success_query[final_new_success] = args.queries
    current_pred = final_pred
    final_l2, final_linf = patch_l2_linf_batch(x_batch, patches, locs)

    rows: List[Dict[str, object]] = []
    for idx in range(batch_size):
        success_query = (
            int(first_success_query[idx].item())
            if int(first_success_query[idx].item()) > 0 and bool(final_success[idx].item())
            else None
        )
        save_result(
            save_prefixes[idx],
            x_orig=x_batch[idx],
            x_adv=x_adv[idx],
            adversarial=bool(final_success[idx].item()),
            untargeted_success=bool(final_adv[idx].item()),
            targeted_success=bool(final_targeted[idx].item()),
            queries=args.queries,
            loc=locs[idx],
            initial_loc=saved_initial_locs[idx],
            patch=patches[idx],
            true_label=int(true_labels[idx].item()),
            target_class=int(target_labels[idx].item()),
            final_prediction=int(current_pred[idx].item()),
            first_success_query=success_query,
            target_probability=float(final_prob[idx].item()),
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
            step_size=args.step_size,
            target_mode=args.target_mode,
            success_mode=args.success_mode,
            save_images=args.save_images,
        )
        rows.append(
            {
                "index": args.current_indices[idx],
                "image_path": args.current_image_paths[idx],
                "output_prefix": str(save_prefixes[idx]),
                "attack": "adversarial_patch",
                "attack_model": attack_model_name,
                "model": getattr(model, "model_name", str(args.model)),
                "model_source": attack_model_source,
                "patch_size": int(patch_size),
                "position_rule": args.position_rule,
                "patch_init": args.patch_init,
                "step_size": float(args.step_size),
                "target_mode": args.target_mode,
                "success_mode": args.success_mode,
                "true_label": int(true_labels[idx].item()),
                "target_class": int(target_labels[idx].item()),
                "adversarial": int(bool(final_success[idx].item())),
                "untargeted_success": int(bool(final_adv[idx].item())),
                "targeted_success": int(bool(final_targeted[idx].item())),
                "target_probability": float(final_prob[idx].item()),
                "first_success_query": "" if success_query is None else success_query,
                "final_prediction": int(current_pred[idx].item()),
                "queries": int(args.queries),
                "loc_y": int(locs[idx, 0].item()),
                "loc_x": int(locs[idx, 1].item()),
                "patch_position_y": int(locs[idx, 0].item()),
                "patch_position_x": int(locs[idx, 1].item()),
                "patch_position_h": int(patch_size),
                "patch_position_w": int(patch_size),
                "initial_loc_y": int(saved_initial_locs[idx][0]),
                "initial_loc_x": int(saved_initial_locs[idx][1]),
                "fixed_location": int(args.fixed_position),
                "location_source": args.location_source or "",
                "eps_linf": "" if args.linf is None else float(args.linf),
                "final_l2": float(final_l2[idx]),
                "final_linf": float(final_linf[idx]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run targeted Adversarial Patch Attack on CSV images.",
    )
    parser.add_argument("--images_csv", "--images-csv", dest="images_csv", required=True)
    parser.add_argument("--save_root", "--save-root", dest="save_root", required=True)
    parser.add_argument("--model", default="1")
    parser.add_argument("--model_source", choices=("auto", "bcos", "torchvision"), default="bcos")
    parser.add_argument("--s", type=int, default=16)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--li", type=int, default=4)
    parser.add_argument("--linf", type=parse_linf, default=None)
    parser.add_argument("--step_size", "--step-size", dest="step_size", type=parse_step_size, default=parse_step_size("1/256"))
    parser.add_argument("--gradient_mode", "--gradient-mode", dest="gradient_mode", choices=("sign", "raw"), default="sign")
    parser.add_argument("--patch_init", "--patch-init", dest="patch_init", choices=PATCH_INITS, default="random_linf")
    parser.add_argument("--target_class", "--target-class", dest="target_class", type=int, default=859)
    parser.add_argument("--target_mode", "--target-mode", dest="target_mode", choices=TARGET_MODES, default="fixed")
    parser.add_argument("--success_mode", "--success-mode", dest="success_mode", choices=SUCCESS_MODES, default="untargeted")
    parser.add_argument("--probability_threshold", "--probability-threshold", dest="probability_threshold", type=float, default=0.9)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--image_batch_size", "--image-batch-size", dest="image_batch_size", type=int, default=16)
    parser.add_argument("--limit_images", "--limit-images", dest="limit_images", type=int, default=0)
    parser.add_argument("--position_rule", "--position-rule", dest="position_rule", choices=POSITION_RULES, default="random")
    parser.add_argument("--fixed_position", "--fixed-position", dest="fixed_position", action="store_true")
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
    if args.queries <= 0:
        parser.error("--queries must be > 0")
    if not args.fixed_position and abs(int(args.li)) <= 1:
        parser.error("--li must have abs(value) > 1 when movement is enabled")
    if not (0.0 <= float(args.probability_threshold) <= 1.0):
        parser.error("--probability_threshold must be in [0, 1]")
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
    print(
        "Algorithm: A-LinCui targeted Adversarial Patch; "
        f"position_rule={args.position_rule}; fixed_position={args.fixed_position}; "
        f"target_mode={args.target_mode}; target_class={args.target_class}; "
        f"success_mode={args.success_mode}; patch_init={args.patch_init}; "
        f"step_size={args.step_size}"
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
            save_root / f"{indices[idx]:05d}_{stems[idx]}_label_{int(true_labels[idx].item())}_adversarial_patch"
            for idx in range(len(chunk))
        ]

        print(f"\nBatch images {start + 1}-{start + len(chunk)}/{len(items)}")
        x_batch, x_chw_tensors = load_image_batch(image_paths, load_image, device)

        initial_locs = None
        args.location_source = "random_fixed" if (args.position_rule == "random" and args.fixed_position) else "random_init"
        if args.position_rule != "random":
            args.location_source = bcos_location_source(args.position_rule, args.fixed_position)
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
        rows = run_adversarial_patch_batch(
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
        targeted = sum(int(row["targeted_success"]) for row in all_rows)
        print(
            f"Summary so far: {successes}/{len(all_rows)} successful attacks; "
            f"targeted={targeted}/{len(all_rows)}"
        )

    print(f"\nDone. Summary: {save_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
