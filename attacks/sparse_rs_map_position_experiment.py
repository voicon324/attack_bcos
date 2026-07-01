from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
import zipfile
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTACKS_DIR = PROJECT_ROOT / "attacks"
for path in (PROJECT_ROOT, ATTACKS_DIR, PROJECT_ROOT / "B-cos-v2"):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bcos  # noqa: E402
from bcos_es_patch import extract_attribution, load_rgb_image, to_bcos_input  # noqa: E402
from sparse_attack import configure_fast_runtime, load_image_paths_from_csv, maybe_channels_last  # noqa: E402


MODEL_ALIASES = {
    "resnet18": "resnet_18",
    "resnet50": "resnet_50",
    "densenet121": "densenet_121",
    "convnext_tiny": "convnext_tiny",
    "convnext_base": "convnext_base",
    "vitc_s": "vitc_s",
    "vitc_b": "vitc_b",
}


@dataclass(frozen=True)
class Arm:
    name: str
    sampler: str
    temperature: float | None


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_temperature(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized in {"inf", "infinity", "uniform"}:
        return None
    out = float(value)
    if out <= 0:
        raise ValueError("temperature must be positive or inf")
    return out


def true_label_from_path(path: str) -> int | None:
    parent = Path(path.replace("\\", "/")).parent.name
    return int(parent) if parent.isdigit() else None


def model_forward(model: torch.nn.Module, x_rgb: Tensor, channels_last: bool) -> Tensor:
    x_bcos = maybe_channels_last(to_bcos_input(x_rgb), channels_last)
    return model(x_bcos)


def random_choice(shape: Iterable[int], device: torch.device) -> Tensor:
    return torch.sign(2 * torch.rand(tuple(shape), device=device) - 1).clamp(0.0, 1.0)


def margin_and_loss(model: torch.nn.Module, x: Tensor, y: Tensor, channels_last: bool) -> tuple[Tensor, Tensor, Tensor]:
    logits = model_forward(model, x, channels_last)
    xent = F.cross_entropy(logits, y, reduction="none")
    row_idx = torch.arange(x.shape[0], device=logits.device)
    y_corr = logits[row_idx, y].clone()
    masked = logits.clone()
    masked[row_idx, y] = -float("inf")
    y_others = masked.max(dim=-1).values
    margin = y_corr - y_others
    return margin, margin, logits


def p_selection(it: int, n_queries: int, p_init: float, rescale_schedule: bool, constant_schedule: bool) -> float:
    if rescale_schedule:
        it = int(it / n_queries * 10000)
    if 0 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 5
    elif 500 < it <= 1000:
        p = p_init / 6
    elif 1000 < it <= 2000:
        p = p_init / 8
    elif 2000 < it <= 4000:
        p = p_init / 10
    elif 4000 < it <= 6000:
        p = p_init / 12
    elif 6000 < it <= 8000:
        p = p_init / 15
    elif 8000 < it:
        p = p_init / 20
    else:
        p = p_init
    return p_init / 2 if constant_schedule else p


def normalize_importance(scores: Tensor, temperature: float | None, eps: float = 1e-12) -> Tensor:
    flat = scores.detach().float().flatten().clamp_min(0)
    if temperature is None or not torch.isfinite(torch.tensor(float(temperature))):
        return torch.ones_like(flat) / max(1, flat.numel())
    if float(flat.max().item()) <= 0:
        return torch.ones_like(flat) / max(1, flat.numel())
    flat = flat + eps
    logits = torch.log(flat) / float(temperature)
    logits = logits - logits.max()
    probs = torch.exp(logits)
    total = probs.sum()
    if not torch.isfinite(total) or float(total.item()) <= 0:
        return torch.ones_like(flat) / max(1, flat.numel())
    return probs / total


def weighted_sample_without_replacement(
    candidate_indices: Tensor,
    base_probs: Tensor,
    k: int,
    generator: torch.Generator,
) -> Tensor:
    if candidate_indices.numel() == 0 or k <= 0:
        return candidate_indices.new_empty((0,), dtype=torch.long)
    k = min(int(k), int(candidate_indices.numel()))
    weights = base_probs[candidate_indices].clone().float()
    if float(weights.sum().item()) <= 0 or not torch.isfinite(weights).all():
        weights = torch.ones_like(weights)
    choice = torch.multinomial(weights, num_samples=k, replacement=False, generator=generator)
    return candidate_indices[choice]


def complement_indices(n_pixels: int, selected: Tensor, device: torch.device) -> Tensor:
    mask = torch.ones(n_pixels, dtype=torch.bool, device=device)
    if selected.numel():
        mask[selected] = False
    return torch.nonzero(mask, as_tuple=False).flatten()


def bcos_contribution_map(model: torch.nn.Module, x_rgb: Tensor, target_class: int) -> Tensor:
    attr = extract_attribution(model, to_bcos_input(x_rgb), target_class)
    cmap = attr["contribution_map"].detach()
    while cmap.dim() > 2:
        cmap = cmap.squeeze(0)
    if cmap.dim() == 3:
        cmap = cmap.abs().sum(dim=0)
    else:
        cmap = cmap.abs()
    if cmap.shape[-2:] != x_rgb.shape[-2:]:
        cmap = F.interpolate(cmap.view(1, 1, *cmap.shape[-2:]), size=x_rgb.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return cmap.detach().clamp_min(0)


def build_probs_for_arm(
    arm: Arm,
    map_scores: Tensor | None,
    n_pixels: int,
    seed: int,
    device: torch.device,
) -> tuple[Tensor, str, float | None, int]:
    if arm.sampler == "uniform" or map_scores is None:
        return torch.ones(n_pixels, device=device) / n_pixels, "uniform", None, 0
    flat = map_scores.flatten().to(device)
    if arm.sampler == "map_shuffle":
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        flat = flat[torch.randperm(flat.numel(), generator=gen, device=device)]
        location_source = "bcos_top1_permuted"
    elif arm.sampler == "map_true":
        location_source = "bcos_top1_true"
    else:
        raise ValueError(f"Unsupported sampler: {arm.sampler}")
    return normalize_importance(flat, arm.temperature).to(device), location_source, arm.temperature, 1


def stable_int_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16)


def run_l0_sparse_rs(
    model: torch.nn.Module,
    x: Tensor,
    true_label: int,
    probs: Tensor,
    n_queries: int,
    eps_pixels: int,
    seed: int,
    channels_last: bool,
    p_init: float,
    rescale_schedule: bool,
    constant_schedule: bool,
) -> dict[str, Any]:
    device = x.device
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    y = torch.tensor([int(true_label)], dtype=torch.long, device=device)

    with torch.no_grad():
        clean_logits = model_forward(model, x, channels_last)
        clean_pred = int(clean_logits.argmax(dim=1).item())
        clean_margin, _, _ = margin_and_loss(model, x, y, channels_last)
        if clean_pred != int(true_label):
            return {
                "clean_prediction": clean_pred,
                "clean_correct": 0,
                "adversarial": 1,
                "first_success_query": 0,
                "queries_used": 0,
                "final_prediction": clean_pred,
                "final_margin": float(clean_margin.item()),
                "changed_pixels": 0,
                "accepted_updates": 0,
            }

        batch_size, channels, height, width = x.shape
        assert batch_size == 1
        n_pixels = height * width
        eps_pixels = min(int(eps_pixels), n_pixels)

        x_best = x.clone()
        ind_p = weighted_sample_without_replacement(torch.arange(n_pixels, device=device), probs, eps_pixels, gen)
        ind_np = complement_indices(n_pixels, ind_p, device)
        x_best[0, :, ind_p // width, ind_p % width] = random_choice([channels, eps_pixels], device)

        margin_min, loss_min, logits = margin_and_loss(model, x_best, y, channels_last)
        first_success_query = 1 if float(margin_min.item()) <= 0 else None
        queries_used = 1
        accepted_updates = 0

        for it in range(1, int(n_queries)):
            if float(margin_min.item()) <= 0:
                break
            eps_it = max(int(p_selection(it, n_queries, p_init, rescale_schedule, constant_schedule) * eps_pixels), 1)
            remove_pos = torch.randperm(eps_pixels, generator=gen, device=device)[:eps_it]
            add_pos = weighted_sample_without_replacement(ind_np, probs, eps_it, gen)
            if add_pos.numel() == 0:
                break

            x_new = x_best.clone()
            p_set = ind_p[remove_pos]
            np_set = add_pos
            x_new[0, :, p_set // width, p_set % width] = x[0, :, p_set // width, p_set % width].clone()
            if eps_it > 1:
                x_new[0, :, np_set // width, np_set % width] = random_choice([channels, np_set.numel()], device)
            else:
                old_clr = x_new[0, :, np_set // width, np_set % width].clone()
                new_clr = old_clr.clone()
                tries = 0
                while (new_clr == old_clr).all().item() and tries < 32:
                    new_clr = random_choice([channels, 1], device)
                    tries += 1
                x_new[0, :, np_set // width, np_set % width] = new_clr

            margin, loss, logits = margin_and_loss(model, x_new, y, channels_last)
            queries_used += 1
            improved = bool(float(loss.item()) < float(loss_min.item()))
            misclassified = bool(float(margin.item()) <= 0)
            if improved or misclassified:
                x_best = x_new
                margin_min = margin.detach()
                loss_min = loss.detach()
                accepted_updates += 1

                old_ind_p = ind_p.clone()
                old_ind_np = ind_np.clone()
                ind_p = old_ind_p.clone()
                ind_np = old_ind_np.clone()
                swap_count = min(remove_pos.numel(), np_set.numel())
                ind_p[remove_pos[:swap_count]] = np_set[:swap_count]
                replace_mask = torch.isin(ind_np, np_set[:swap_count])
                ind_np[replace_mask] = p_set[: int(replace_mask.sum().item())]

                if misclassified and first_success_query is None:
                    first_success_query = queries_used
                    break

        with torch.no_grad():
            final_logits = model_forward(model, x_best, channels_last)
            final_pred = int(final_logits.argmax(dim=1).item())
            final_margin, _, _ = margin_and_loss(model, x_best, y, channels_last)
        return {
            "clean_prediction": clean_pred,
            "clean_correct": 1,
            "adversarial": int(final_pred != int(true_label)),
            "first_success_query": "" if first_success_query is None else int(first_success_query),
            "queries_used": int(queries_used),
            "final_prediction": final_pred,
            "final_margin": float(final_margin.item()),
            "changed_pixels": int(ind_p.numel()),
            "accepted_updates": int(accepted_updates),
        }


def arm_list(temperatures: list[float | None]) -> list[Arm]:
    arms = [Arm("uniform", "uniform", None)]
    for temp in temperatures:
        if temp is None:
            continue
        temp_slug = str(temp).replace(".", "p")
        arms.append(Arm(f"map_shuffle_tau_{temp_slug}", "map_shuffle", temp))
        arms.append(Arm(f"map_true_tau_{temp_slug}", "map_true", temp))
    return arms


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], *, clean_only: bool) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if clean_only and int(row["clean_correct"]) != 1:
            continue
        groups[(row["arm"], row["sampler"], row["temperature"])].append(row)
    out = []
    for key, group in sorted(groups.items()):
        successes = [row for row in group if int(row["adversarial"]) == 1]
        queries = [int(row["effective_first_success_query"]) for row in successes if str(row["effective_first_success_query"]) != ""]
        out.append({
            "arm": key[0],
            "sampler": key[1],
            "temperature": key[2],
            "images": len(group),
            "clean_correct_images": sum(int(row["clean_correct"]) for row in group),
            "successes": len(successes),
            "success_rate": len(successes) / len(group) if group else "",
            "median_effective_first_success_query": float(np.median(queries)) if queries else "",
            "mean_effective_first_success_query": float(np.mean(queries)) if queries else "",
        })
    return out


def success_by_query(rows: list[dict[str, Any]], *, clean_only: bool) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if clean_only and int(row["clean_correct"]) != 1:
            continue
        grouped[(row["arm"], row["sampler"], row["temperature"])].append(row)
    out = []
    for key, group in sorted(grouped.items()):
        events: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            query = row.get("effective_first_success_query", "")
            if query != "":
                events[int(query)].append(row)
        cumulative = 0
        for query in sorted(events):
            cumulative += len(events[query])
            out.append({
                "arm": key[0],
                "sampler": key[1],
                "temperature": key[2],
                "first_success_query": query,
                "new_successes": len(events[query]),
                "cumulative_successes": cumulative,
                "denominator_images": len(group),
                "success_rate": cumulative / len(group) if group else "",
                "image_indices": ";".join(str(row["image_index"]) for row in events[query]),
            })
    return out


def zip_result(zip_path: Path, result_dir: Path, extra_files: list[Path]) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(result_dir.rglob("*")):
            if path.is_file():
                archive.write(path, Path("outputs") / path.relative_to(result_dir))
        for path in extra_files:
            if path.is_file():
                archive.write(path, path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparse-RS L0 map-position fairness experiment.")
    parser.add_argument("--images-csv", required=True)
    parser.add_argument("--save-root", required=True)
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--eps-pixels", type=int, default=64)
    parser.add_argument("--limit-images", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--temperatures", nargs="+", default=["4", "1", "0.25"])
    parser.add_argument("--p-init", type=float, default=0.8)
    parser.add_argument("--constant-schedule", action="store_true")
    parser.add_argument("--no-rescale-schedule", action="store_true")
    parser.add_argument("--zip-path", type=Path, default=None)
    parser.add_argument("--run-log", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    configure_fast_runtime(device)
    channels_last = device.type == "cuda"
    result_dir = Path(args.save_root).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    model_name = MODEL_ALIASES.get(args.model, args.model)
    model_fn = getattr(bcos.pretrained, model_name)
    model = model_fn(pretrained=True).to(device).eval()
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    temperatures = [parse_temperature(value) for value in args.temperatures]
    arms = arm_list(temperatures)
    image_items = load_image_paths_from_csv(Path(args.images_csv))
    if args.limit_images > 0:
        image_items = image_items[: args.limit_images]

    rows: list[dict[str, Any]] = []
    print(f"model={args.model} images={len(image_items)} queries={args.queries} eps_pixels={args.eps_pixels}")
    print("arms=" + ",".join(arm.name for arm in arms))
    for image_num, (image_index, image_path) in enumerate(image_items, start=1):
        x_rgb, resolved_path = load_rgb_image(image_path, device=device)
        x_rgb = maybe_channels_last(x_rgb, channels_last)
        true_label = true_label_from_path(resolved_path)
        if true_label is None:
            raise ValueError(f"Cannot infer true label from image path: {resolved_path}")
        with torch.no_grad():
            clean_logits = model_forward(model, x_rgb, channels_last)
            clean_pred = int(clean_logits.argmax(dim=1).item())
        map_scores = bcos_contribution_map(model, x_rgb, clean_pred)
        map_entropy = float((-(normalize_importance(map_scores.flatten(), 1.0) * torch.log(normalize_importance(map_scores.flatten(), 1.0) + 1e-12))).sum().item())
        print(f"[{image_num}/{len(image_items)}] index={image_index} true={true_label} pred={clean_pred}")
        for seed in args.seeds:
            for arm in arms:
                arm_seed = int(seed) * 1000003 + int(image_index or image_num) * 9176 + stable_int_hash(arm.name) % 9973
                probs, location_source, temperature, greybox_probe_queries = build_probs_for_arm(
                    arm, map_scores, x_rgb.shape[-1] * x_rgb.shape[-2], arm_seed, device
                )
                result = run_l0_sparse_rs(
                    model=model,
                    x=x_rgb,
                    true_label=true_label,
                    probs=probs,
                    n_queries=args.queries,
                    eps_pixels=args.eps_pixels,
                    seed=arm_seed,
                    channels_last=channels_last,
                    p_init=args.p_init,
                    rescale_schedule=not args.no_rescale_schedule,
                    constant_schedule=args.constant_schedule,
                )
                fsq = result["first_success_query"]
                effective = "" if fsq == "" else (0 if int(fsq) == 0 else int(fsq) + int(greybox_probe_queries))
                row = {
                    "attack": "sparse_rs_map_l0",
                    "model": args.model,
                    "model_source": "bcos",
                    "image_index": image_index,
                    "image_path": resolved_path,
                    "true_label": true_label,
                    "seed": seed,
                    "arm": arm.name,
                    "sampler": arm.sampler,
                    "temperature": "" if temperature is None else temperature,
                    "eps_pixels": args.eps_pixels,
                    "queries": args.queries,
                    "greybox_probe_queries": greybox_probe_queries,
                    "location_source": location_source,
                    "map_entropy": map_entropy,
                    "patch_position_y": "",
                    "patch_position_x": "",
                    "patch_position_h": "",
                    "patch_position_w": "",
                    **result,
                    "effective_first_success_query": effective,
                }
                rows.append(row)
                print(
                    f"  {arm.name} seed={seed} adv={row['adversarial']} "
                    f"q={row['first_success_query']} eff_q={row['effective_first_success_query']}"
                )
                write_csv(result_dir / "summary.csv", rows)

    write_csv(result_dir / "summary.csv", rows)
    write_csv(result_dir / "summary_all_images.csv", summarize(rows, clean_only=False))
    write_csv(result_dir / "summary_clean_correct.csv", summarize(rows, clean_only=True))
    write_csv(result_dir / "success_by_query.csv", success_by_query(rows, clean_only=False))
    write_csv(result_dir / "success_by_query_clean_correct.csv", success_by_query(rows, clean_only=True))
    success_events = [row for row in rows if int(row["adversarial"]) == 1]
    write_csv(result_dir / "success_events.csv", success_events)
    manifest = {
        "attack": "sparse_rs_map_l0",
        "generated_at": now_iso(),
        "elapsed_sec": round(time.time() - started, 3),
        "model": args.model,
        "images": len(image_items),
        "queries": args.queries,
        "eps_pixels": args.eps_pixels,
        "seeds": args.seeds,
        "temperatures": args.temperatures,
        "arms": [arm.__dict__ for arm in arms],
        "threat_model_note": "map_true and map_shuffle are grey-box arms with one B-cos explanation probe; effective_first_success_query adds greybox_probe_queries.",
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    zip_path = args.zip_path or result_dir.with_suffix(".zip")
    zip_result(zip_path, result_dir, [result_dir / "manifest.json"])
    print(f"Result zip: {zip_path}")


if __name__ == "__main__":
    main()
