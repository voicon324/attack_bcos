"""
Batch ES Patch Attack Experiment on B-cos Models
=================================================

Runs adversarial patch attacks using Evolution Strategies across:
  - Multiple B-cos pretrained models
  - Multiple patch sizes (128, 64, 32)
  - 100 ImageNet validation images

Optimal patch position is found via 2D prefix-sum (DP) on two B-cos
signals: the contribution map and the target-class explanation map.
Each ES candidate is scored at both positions and keeps the better one,
while ES still optimizes only the perturbation.

Usage
-----
    # Full experiment
    python bcos_es_patch_sweep.py

    # Quick test
    python bcos_es_patch_sweep.py --num-images 2 --models resnet18 --patch-sizes 32 --generations 10
"""

import sys
import os
import argparse
import csv
import time
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from attack_utils import canonicalize_norm, project_perturbation

# ── Make sure bcos is importable ────────────────────────────────────────────
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
from bcos.common import explanation_mode

IMAGENET_ROOT = "/kaggle/input/imagenet1kvalid"

# All B-cos models (non-standard, non-ViT for consistency)
ALL_BCOS_MODELS = [
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "resnext50_32x4d",
    "densenet121", "densenet161", "densenet169", "densenet201",
    "vgg11_bnu",
    "convnext_tiny", "convnext_base",
]

DEFAULT_PATCH_SIZES = [128, 64, 32, 16, 8, 4]


# ═════════════════════════════════════════════════════════════════════════════
# 1. UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def to_bcos_input(x_rgb: Tensor) -> Tensor:
    """Converts 3-channel RGB [0,1] to 6-channel B-cos input (x, 1-x)."""
    return torch.cat([x_rgb, 1.0 - x_rgb], dim=1)


def apply_patch(x_rgb: Tensor, patch: Tensor, pos_y: int, pos_x: int) -> Tensor:
    """Adds a patch onto the 3-channel RGB image x_rgb."""
    x_pert = x_rgb.clone()
    _, _, hp, wp = patch.shape
    x_pert[..., pos_y:pos_y+hp, pos_x:pos_x+wp] = torch.clamp(
        x_pert[..., pos_y:pos_y+hp, pos_x:pos_x+wp] + patch, 0.0, 1.0
    )
    return x_pert


def load_rgb_image(image_path: str, device: torch.device) -> Tensor:
    """Load and preprocess a single image to [1, 3, 224, 224] tensor."""
    from PIL import Image
    from torchvision import transforms as T

    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor()
    ])
    return transform(img).unsqueeze(0).to(device)


def collect_image_paths(root: str, num_images: int, seed: int = 42) -> List[str]:
    """Randomly sample `num_images` from ImageNet validation set."""
    all_images = []
    for class_dir in sorted(os.listdir(root)):
        class_path = os.path.join(root, class_dir)
        if not os.path.isdir(class_path):
            continue
        for fname in sorted(os.listdir(class_path)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.JPEG')):
                all_images.append(os.path.join(class_path, fname))

    rng = random.Random(seed)
    rng.shuffle(all_images)
    selected = all_images[:num_images]
    print(f"  Collected {len(all_images)} images total, selected {len(selected)}")
    return selected


# ═════════════════════════════════════════════════════════════════════════════
# 2. OPTIMAL PATCH POSITION VIA 2D PREFIX-SUM
# ═════════════════════════════════════════════════════════════════════════════

def find_best_patch_position_from_importance_map(
    importance_map: np.ndarray,
    patch_size: int,
) -> Tuple[int, int]:
    attr = np.abs(importance_map).astype(np.float64)
    H, W = attr.shape
    S = min(patch_size, H, W)
    prefix = np.zeros((H + 1, W + 1), dtype=np.float64)
    prefix[1:, 1:] = np.cumsum(np.cumsum(attr, axis=0), axis=1)
    best_sum = -1.0
    best_y, best_x = 0, 0
    for y in range(H - S + 1):
        for x in range(W - S + 1):
            window_sum = (
                prefix[y + S, x + S]
                - prefix[y, x + S]
                - prefix[y + S, x]
                + prefix[y, x]
            )
            if window_sum > best_sum:
                best_sum = window_sum
                best_y, best_x = y, x

    return best_y, best_x


def extract_guidance_maps(
    model: torch.nn.Module,
    x_rgb: Tensor,
    target_class: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_6ch = to_bcos_input(x_rgb)
    inp = x_6ch.detach().clone().requires_grad_(True)
    with explanation_mode(model):
        result = model.explain(inp, idx=target_class)

    cmap = result["contribution_map"]
    if cmap.dim() == 4:
        cmap = cmap.squeeze(0)
    if cmap.dim() == 3:
        cmap = cmap.squeeze(0)
    contribution_map = cmap.detach().cpu().numpy()

    explanation = result["explanation"]
    if explanation.shape[-1] >= 4:
        explanation_map = explanation[..., 3]
    else:
        explanation_map = np.abs(explanation).sum(axis=-1)
    return contribution_map, explanation_map


def find_optimal_patch_positions(
    model: torch.nn.Module,
    x_rgb: Tensor,
    target_class: int,
    patch_size: int,
) -> List[Tuple[int, int]]:
    contribution_map, explanation_map = extract_guidance_maps(
        model,
        x_rgb,
        target_class,
    )
    return [
        find_best_patch_position_from_importance_map(contribution_map, patch_size),
        find_best_patch_position_from_importance_map(explanation_map, patch_size),
    ]


# ═════════════════════════════════════════════════════════════════════════════
# 3. SIMPLIFIED ES OPTIMIZER (FIXED POSITION)
# ═════════════════════════════════════════════════════════════════════════════

class FixedPosPatchES:
    """Evolution Strategy optimizer for patch perturbation only (no position search)."""

    def __init__(
        self,
        param_shape: Tuple[int, ...],
        epsilon: float,
        norm: str,
        population_size: int = 50,
        elite_fraction: float = 0.2,
        sigma: float = 0.05,
        sigma_decay: float = 0.99,
        sigma_min: float = 1e-4,
        learning_rate: float = 1.0,
        device: torch.device = torch.device('cpu'),
    ):
        self.param_shape = param_shape
        self.population_size = population_size
        self.elite_size = max(1, int(population_size * elite_fraction))
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min
        self.lr = learning_rate
        self.device = device
        self.epsilon = epsilon
        self.norm = canonicalize_norm(norm)
        self.perturbation = torch.zeros(param_shape, device=device)

    def ask(self) -> Tensor:
        noise = torch.randn(self.population_size, *self.param_shape, device=self.device)
        candidates = self.perturbation.unsqueeze(0) + self.sigma * noise
        return project_perturbation(candidates, self.epsilon, self.norm)

    def tell(self, candidates: Tensor, scores: Tensor) -> Tensor:
        _, elite_idx = scores.topk(self.elite_size)
        elite = candidates[elite_idx]
        elite_scores = scores[elite_idx]
        elite_scores = (elite_scores - elite_scores.mean()) / (elite_scores.std() + 1e-8)
        weights = F.softmax(elite_scores, dim=0)

        w = weights.clone()
        for _ in range(len(self.param_shape)):
            w = w.unsqueeze(-1)

        new_patch = (w * elite).sum(dim=0)
        self.perturbation = (1 - self.lr) * self.perturbation + self.lr * new_patch
        self.perturbation = project_perturbation(self.perturbation, self.epsilon, self.norm)
        self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)
        return self.perturbation

    def get_patch(self) -> Tensor:
        return self.perturbation.clone()


# ═════════════════════════════════════════════════════════════════════════════
# 4. SCORE COMPUTATION (BATCH, FIXED POSITION)
# ═════════════════════════════════════════════════════════════════════════════

def compute_scores_fixed_multi_position(
    model: torch.nn.Module,
    x_rgb: Tensor,
    candidates: Tensor,
    patch_positions: List[Tuple[int, int]],
    target_class: int,
) -> Tuple[Tensor, Tensor]:
    """
    Evaluate candidate patches at multiple fixed positions.
    Returns the best CW margin loss per candidate and the winning position index.
    """
    N = candidates.shape[0]
    position_count = len(patch_positions)
    batch_size = min(N, max(1, 50 // max(position_count, 1)))
    scores_list = []
    pos_idx_list = []
    ps = candidates.shape[-1]

    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = candidates[i:i+batch_size]
            B = batch.shape[0]
            x_rep = x_rgb.expand(B * position_count, -1, -1, -1).clone()
            patch_vals = batch.squeeze(1)
            for pos_idx, (pos_y, pos_x) in enumerate(patch_positions):
                lo = pos_idx * B
                hi = lo + B
                x_rep[lo:hi, :, pos_y:pos_y+ps, pos_x:pos_x+ps] = torch.clamp(
                    x_rep[lo:hi, :, pos_y:pos_y+ps, pos_x:pos_x+ps] + patch_vals, 0.0, 1.0
                )

            x_6ch = to_bcos_input(x_rep)
            outputs = model(x_6ch)

            out_clone = outputs.clone()
            out_clone[:, target_class] = -float('inf')
            max_other = out_clone.max(dim=1)[0]
            cw = (max_other - outputs[:, target_class]).view(position_count, B)
            best_scores, best_pos_idx = cw.max(dim=0)

            scores_list.append(best_scores)
            pos_idx_list.append(best_pos_idx.to(dtype=torch.long))

    return torch.cat(scores_list), torch.cat(pos_idx_list)


# ═════════════════════════════════════════════════════════════════════════════
# 5. SINGLE ATTACK
# ═════════════════════════════════════════════════════════════════════════════

def run_single_attack(
    model: torch.nn.Module,
    x_rgb: Tensor,
    target_class: int,
    patch_size: int,
    patch_positions: List[Tuple[int, int]],
    generations: int = 200,
    population_size: int = 50,
    epsilon: float = 0.3,
    norm: str = "linf",
    sigma: float = 0.05,
    sigma_decay: float = 0.99,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, Any]:
    """Run ES attack with fixed position, return result dict."""

    S = min(patch_size, x_rgb.shape[2], x_rgb.shape[3])

    es = FixedPosPatchES(
        param_shape=(1, 3, S, S),
        epsilon=epsilon,
        norm=norm,
        population_size=population_size,
        sigma=sigma,
        sigma_decay=sigma_decay,
        device=device,
    )

    best_patch_overall: Optional[Tensor] = None
    best_score_overall = -float("inf")
    best_pos_overall = patch_positions[0]

    for gen in range(generations):
        cands = es.ask()
        scores, best_pos_idx = compute_scores_fixed_multi_position(
            model,
            x_rgb,
            cands,
            patch_positions,
            target_class,
        )
        es.tell(cands, scores)
        best_idx = scores.argmax().item()
        best_score = scores[best_idx].item()
        if best_score > best_score_overall:
            best_score_overall = best_score
            best_patch_overall = cands[best_idx].clone()
            best_pos_overall = patch_positions[int(best_pos_idx[best_idx].item())]

    final_patch = project_perturbation(es.get_patch(), epsilon, norm)
    if best_patch_overall is not None:
        final_patch = project_perturbation(best_patch_overall, epsilon, norm)
    _, final_pos_idx = compute_scores_fixed_multi_position(
        model,
        x_rgb,
        final_patch.unsqueeze(0),
        patch_positions,
        target_class,
    )
    final_pos_y, final_pos_x = patch_positions[int(final_pos_idx[0].item())]
    if best_patch_overall is not None:
        final_pos_y, final_pos_x = best_pos_overall
    x_pert = apply_patch(x_rgb, final_patch, final_pos_y, final_pos_x)

    with torch.no_grad():
        out_pert = model(to_bcos_input(x_pert))
        pred_class_pert = out_pert.argmax(1).item()
        logit_pert_target = out_pert[0, target_class].item()
        logit_pert_pred = out_pert[0, pred_class_pert].item()

    return {
        "pert_class": pred_class_pert,
        "pert_logit_target": logit_pert_target,
        "pert_logit_pred": logit_pert_pred,
        "attack_success": pred_class_pert != target_class,
        "patch_pos_y": final_pos_y,
        "patch_pos_x": final_pos_x,
        "x_pert": x_pert,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 6. MAIN EXPERIMENT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Batch ES Patch Attack on B-cos Models")
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Model names (default: all B-cos models)")
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=None,
                        help="Patch sizes (default: 128 64 32)")
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.3,
        help="Perturbation budget. For l0, values <= 1 are treated as a fraction of patch coefficients.",
    )
    parser.add_argument("--norm", type=str, default="linf", help="Perturbation norm budget: l0, l2, or linf")
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--sigma-decay", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default=str(RESULT_DIR / "es_patch_results.csv"))
    parser.add_argument("--save-images", action="store_true", help="Save perturbed images")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model_names = args.models if args.models else ALL_BCOS_MODELS
    patch_sizes = args.patch_sizes if args.patch_sizes else DEFAULT_PATCH_SIZES
    norm_name = canonicalize_norm(args.norm)
    epsilon_key = f"{args.epsilon:g}"
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    perturbed_images_dir = output_path.parent / "perturbed_images"

    print("=" * 72)
    print("  Batch ES Patch Attack Experiment on B-cos Models")
    print("=" * 72)
    print(f"  Device       : {device}")
    print(f"  Models       : {model_names}")
    print(f"  Patch sizes  : {patch_sizes}")
    print(f"  Images       : {args.num_images}")
    print(f"  Generations  : {args.generations}")
    print(f"  Population   : {args.population}")
    print(f"  Budget       : {norm_name} <= {epsilon_key}")
    print(f"  Output CSV   : {output_path}")
    print(f"  Save images  : {args.save_images}")
    print()

    # Collect images
    print("▸ Collecting images …")
    image_paths = collect_image_paths(IMAGENET_ROOT, args.num_images, seed=args.seed)

    if args.save_images:
        perturbed_images_dir.mkdir(parents=True, exist_ok=True)
        import torchvision.utils as vutils

    # Prepare CSV
    csv_fields = [
        "image_path", "model_name", "patch_size",
        "norm", "epsilon",
        "orig_class", "orig_logit",
        "pert_class", "pert_logit_target", "pert_logit_pred",
        "attack_success", "patch_pos_y", "patch_pos_x", "time_seconds",
    ]

    # Check if CSV already exists to support resume
    completed = set()
    if output_path.exists():
        with output_path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_norm = canonicalize_norm(row.get("norm") or "linf")
                row_epsilon = row.get("epsilon") or "0.3"
                key = (row["image_path"], row["model_name"], row["patch_size"], row_norm, row_epsilon)
                completed.add(key)
        print(f"  Found existing CSV with {len(completed)} completed entries (resuming)")
        csv_file = output_path.open("a", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    else:
        csv_file = output_path.open("w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()

    total_experiments = len(model_names) * len(image_paths) * len(patch_sizes)
    print(f"  Total experiments: {total_experiments}\n")

    experiment_count = 0
    total_success = 0
    total_done = 0

    # Outer loop: model (load once, reuse for all images)
    for mi, model_name in enumerate(model_names):
        print(f"\n{'='*72}")
        print(f"  Model [{mi+1}/{len(model_names)}]: {model_name}")
        print(f"{'='*72}")

        # Load model
        t_model_start = time.time()
        try:
            model_fn = getattr(bcos.pretrained, model_name)
            model = model_fn(pretrained=True).to(device).eval()
        except Exception as e:
            print(f"  ❌ Failed to load model {model_name}: {e}")
            continue
        t_model_load = time.time() - t_model_start
        print(f"  ✓ Model loaded in {t_model_load:.1f}s")

        for ii, img_path in enumerate(image_paths):
            # Load image
            try:
                x_rgb = load_rgb_image(img_path, device)
            except Exception as e:
                print(f"  ❌ Failed to load image {img_path}: {e}")
                continue

            # Get original prediction
            with torch.no_grad():
                out_orig = model(to_bcos_input(x_rgb))
                orig_class = out_orig.argmax(1).item()
                orig_logit = out_orig[0, orig_class].item()

            for ps in patch_sizes:
                experiment_count += 1
                key = (img_path, model_name, str(ps), norm_name, epsilon_key)
                if key in completed:
                    continue

                # Skip if patch is larger than image
                _, _, H, W = x_rgb.shape
                if ps > H or ps > W:
                    print(f"  ⚠ Patch {ps} > image {H}x{W}, skipping")
                    continue

                t_start = time.time()

                # Find optimal position via DP on explanation map
                try:
                    pos_y, pos_x = find_optimal_patch_position(
                        model, x_rgb, orig_class, ps
                    )
                except Exception as e:
                    print(f"  ❌ Position search failed: {e}")
                    continue

                # Run ES attack
                try:
                    result = run_single_attack(
                        model, x_rgb, orig_class, ps,
                        pos_y, pos_x,
                        generations=args.generations,
                        population_size=args.population,
                        epsilon=args.epsilon,
                        norm=norm_name,
                        sigma=args.sigma,
                        sigma_decay=args.sigma_decay,
                        device=device,
                    )
                except Exception as e:
                    print(f"  ❌ Attack failed: {e}")
                    continue

                elapsed = time.time() - t_start
                total_done += 1
                if result["attack_success"]:
                    total_success += 1

                # Write row
                row = {
                    "image_path": img_path,
                    "model_name": model_name,
                    "patch_size": ps,
                    "norm": norm_name,
                    "epsilon": epsilon_key,
                    "orig_class": orig_class,
                    "orig_logit": f"{orig_logit:.4f}",
                    "pert_class": result["pert_class"],
                    "pert_logit_target": f"{result['pert_logit_target']:.4f}",
                    "pert_logit_pred": f"{result['pert_logit_pred']:.4f}",
                    "attack_success": result["attack_success"],
                    "patch_pos_y": result["patch_pos_y"],
                    "patch_pos_x": result["patch_pos_x"],
                    "time_seconds": f"{elapsed:.2f}",
                }
                writer.writerow(row)
                csv_file.flush()

                if args.save_images and result["attack_success"]:
                    img_name = os.path.basename(img_path).split('.')[0]
                    save_path = perturbed_images_dir / f"{img_name}_{model_name}_p{ps}.png"
                    vutils.save_image(result["x_pert"], save_path)

                status = "✓" if result["attack_success"] else "✗"
                img_basename = os.path.basename(img_path)
                print(
                    f"  [{experiment_count}/{total_experiments}] "
                    f"{status} img={img_basename} ps={ps} "
                    f"logit: {orig_logit:.2f}→{result['pert_logit_target']:.2f} "
                    f"pred: {orig_class}→{result['pert_class']} "
                    f"pos=({pos_y},{pos_x}) "
                    f"t={elapsed:.1f}s", flush=True
                )

        # Free model memory
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    csv_file.close()

    # Print summary
    print(f"\n{'='*72}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"{'='*72}")
    print(f"  Total experiments : {total_done}")
    print(f"  Attack successes  : {total_success}")
    print(f"  Success rate      : {total_success/max(1,total_done)*100:.1f}%")
    print(f"  Results saved to  : {output_path}")

    # Print per-model × per-patch-size summary
    if output_path.exists():
        print(f"\n  {'Model':<25s} {'PatchSize':>10s} {'Success':>8s} {'Total':>6s} {'Rate':>8s}")
        print(f"  {'-'*25} {'-'*10} {'-'*8} {'-'*6} {'-'*8}")

        import collections
        stats = collections.defaultdict(lambda: {"success": 0, "total": 0})
        with output_path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["model_name"], row["patch_size"])
                stats[key]["total"] += 1
                if row["attack_success"] == "True":
                    stats[key]["success"] += 1

        for (mn, ps), s in sorted(stats.items()):
            rate = s["success"] / max(1, s["total"]) * 100
            print(f"  {mn:<25s} {ps:>10s} {s['success']:>8d} {s['total']:>6d} {rate:>7.1f}%")

    print(f"\n{'='*72}")
    print(f"  Done!")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
