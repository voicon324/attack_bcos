"""
Batch ES Patch Attack Experiment on B-cos Models
=================================================

Runs adversarial patch attacks using Evolution Strategies across:
  - Multiple B-cos pretrained models
  - Multiple patch sizes (128, 64, 32)
  - 100 ImageNet validation images

Optimal patch position is found via 2D prefix-sum (DP) on the
B-cos contribution/explanation map (maximizing total attribution
inside the patch window). ES then optimizes only the perturbation.

Usage
-----
    # Full experiment
    python es_patch_experiment.py

    # Quick test
    python es_patch_experiment.py --num-images 2 --models resnet18 --patch-sizes 32 --generations 10
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

# ── Make sure bcos is importable ────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
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

def find_optimal_patch_position(
    model: torch.nn.Module,
    x_rgb: Tensor,
    target_class: int,
    patch_size: int,
) -> Tuple[int, int]:
    """
    Find the patch position that covers the region with maximum total
    explanation attribution, using 2D prefix-sum for O(H*W) search.

    Parameters
    ----------
    model : B-cos model
    x_rgb : [1, 3, H, W] RGB image tensor
    target_class : class to explain
    patch_size : side length of the square patch

    Returns
    -------
    (pos_y, pos_x) : top-left corner of the optimal patch position
    """
    _, _, H, W = x_rgb.shape
    S = min(patch_size, H, W)  # Clamp patch size to image dims

    # Get contribution map from B-cos explanation
    x_6ch = to_bcos_input(x_rgb)
    inp = x_6ch.detach().clone().requires_grad_(True)
    with explanation_mode(model):
        result = model.explain(inp, idx=target_class)

    # contribution_map shape: [1, 1, H, W] or similar
    cmap = result["contribution_map"]
    if cmap.dim() == 4:
        cmap = cmap.squeeze(0)  # [1, H, W]
    if cmap.dim() == 3:
        cmap = cmap.squeeze(0)  # [H, W]

    # Use absolute contribution values — we want to place patch where
    # the model relies most on the input
    attr = cmap.abs().detach().cpu().numpy().astype(np.float64)

    # Build 2D prefix sum using numpy cumsum (vectorized)
    prefix = np.zeros((H + 1, W + 1), dtype=np.float64)
    prefix[1:, 1:] = np.cumsum(np.cumsum(attr, axis=0), axis=1)

    # Find S×S window with maximum sum
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


# ═════════════════════════════════════════════════════════════════════════════
# 3. SIMPLIFIED ES OPTIMIZER (FIXED POSITION)
# ═════════════════════════════════════════════════════════════════════════════

class FixedPosPatchES:
    """Evolution Strategy optimizer for patch perturbation only (no position search)."""

    def __init__(
        self,
        param_shape: Tuple[int, ...],
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
        self.patch = torch.zeros(param_shape, device=device)

    def ask(self) -> Tensor:
        noise = torch.randn(self.population_size, *self.param_shape, device=self.device)
        candidates = self.patch.unsqueeze(0) + self.sigma * noise
        return candidates.clamp(-1.0, 1.0)

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
        self.patch = (1 - self.lr) * self.patch + self.lr * new_patch
        self.patch = self.patch.clamp(-1.0, 1.0)
        self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)
        return self.patch

    def get_patch(self) -> Tensor:
        return self.patch.clone()


# ═════════════════════════════════════════════════════════════════════════════
# 4. SCORE COMPUTATION (BATCH, FIXED POSITION)
# ═════════════════════════════════════════════════════════════════════════════

def compute_scores_fixed(
    model: torch.nn.Module,
    x_rgb: Tensor,
    candidates: Tensor,
    pos_y: int,
    pos_x: int,
    target_class: int,
) -> Tensor:
    """
    Evaluate candidate patches at a fixed position.
    Returns CW margin loss: max(other_logits) - logit[target_class].
    """
    N = candidates.shape[0]
    batch_size = min(N, 50)  # Adjust for GPU memory
    scores_list = []
    _, _, H, W = x_rgb.shape
    # candidates shape: [N, 1, 3, S, S]
    ps = candidates.shape[-1]  # patch spatial size

    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = candidates[i:i+batch_size]
            B = batch.shape[0]
            x_rep = x_rgb.expand(B, -1, -1, -1).clone()

            # batch shape: [B, 1, 3, S, S] -> squeeze to [B, 3, S, S]
            patch_vals = batch.squeeze(1)
            # Apply all patches at once
            x_rep[:, :, pos_y:pos_y+ps, pos_x:pos_x+ps] = torch.clamp(
                x_rep[:, :, pos_y:pos_y+ps, pos_x:pos_x+ps] + patch_vals, 0.0, 1.0
            )

            x_6ch = to_bcos_input(x_rep)
            outputs = model(x_6ch)

            # CW margin loss
            out_clone = outputs.clone()
            out_clone[:, target_class] = -float('inf')
            max_other = out_clone.max(dim=1)[0]
            cw = max_other - outputs[:, target_class]

            scores_list.append(cw)

    return torch.cat(scores_list)


# ═════════════════════════════════════════════════════════════════════════════
# 5. SINGLE ATTACK
# ═════════════════════════════════════════════════════════════════════════════

def run_single_attack(
    model: torch.nn.Module,
    x_rgb: Tensor,
    target_class: int,
    patch_size: int,
    pos_y: int,
    pos_x: int,
    generations: int = 200,
    population_size: int = 50,
    sigma: float = 0.05,
    sigma_decay: float = 0.99,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, Any]:
    """Run ES attack with fixed position, return result dict."""

    S = min(patch_size, x_rgb.shape[2], x_rgb.shape[3])

    es = FixedPosPatchES(
        param_shape=(1, 3, S, S),
        population_size=population_size,
        sigma=sigma,
        sigma_decay=sigma_decay,
        device=device,
    )

    for gen in range(generations):
        cands = es.ask()
        scores = compute_scores_fixed(model, x_rgb, cands, pos_y, pos_x, target_class)
        es.tell(cands, scores)

    # Final evaluation
    final_patch = es.get_patch().clamp(-0.3, 0.3)
    x_pert = apply_patch(x_rgb, final_patch, pos_y, pos_x)

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
        "patch_pos_y": pos_y,
        "patch_pos_x": pos_x,
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
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--sigma-decay", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="es_patch_results.csv")
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

    print("=" * 72)
    print("  Batch ES Patch Attack Experiment on B-cos Models")
    print("=" * 72)
    print(f"  Device       : {device}")
    print(f"  Models       : {model_names}")
    print(f"  Patch sizes  : {patch_sizes}")
    print(f"  Images       : {args.num_images}")
    print(f"  Generations  : {args.generations}")
    print(f"  Population   : {args.population}")
    print(f"  Output CSV   : {args.output}")
    print(f"  Save images  : {args.save_images}")
    print()

    # Collect images
    print("▸ Collecting images …")
    image_paths = collect_image_paths(IMAGENET_ROOT, args.num_images, seed=args.seed)

    if args.save_images:
        os.makedirs("perturbed_images", exist_ok=True)
        import torchvision.utils as vutils

    # Prepare CSV
    csv_fields = [
        "image_path", "model_name", "patch_size",
        "orig_class", "orig_logit",
        "pert_class", "pert_logit_target", "pert_logit_pred",
        "attack_success", "patch_pos_y", "patch_pos_x", "time_seconds",
    ]

    # Check if CSV already exists to support resume
    completed = set()
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["image_path"], row["model_name"], row["patch_size"])
                completed.add(key)
        print(f"  Found existing CSV with {len(completed)} completed entries (resuming)")
        csv_file = open(args.output, "a", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    else:
        csv_file = open(args.output, "w", newline="")
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
                key = (img_path, model_name, str(ps))
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
                    save_path = os.path.join("perturbed_images", f"{img_name}_{model_name}_p{ps}.png")
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
    print(f"  Results saved to  : {args.output}")

    # Print per-model × per-patch-size summary
    if os.path.exists(args.output):
        print(f"\n  {'Model':<25s} {'PatchSize':>10s} {'Success':>8s} {'Total':>6s} {'Rate':>8s}")
        print(f"  {'-'*25} {'-'*10} {'-'*8} {'-'*6} {'-'*8}")

        import collections
        stats = collections.defaultdict(lambda: {"success": 0, "total": 0})
        with open(args.output, "r") as f:
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
