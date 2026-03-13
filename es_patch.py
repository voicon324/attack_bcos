"""
Evolution Strategies (ES) for B-cos Counterfactual Patch Attack
===============================================================

This script uses gradient-free Evolution Strategies to find an adversarial 
patch that maximally decreases the predicted probability of the original label.

Key concepts
------------
* Optimize a localized patch of size S x S.
* Objective: minimize the logit (or probability) of the original predicted class.
* Output: A comprehensive visualization including original and perturbed images,
  the patch, prediction changes, and explanation maps before and after.

Usage
-----
    python es_patch.py --model resnet18 --patch-size 40
    python es_patch.py --image my.jpg --generations 100
"""

import sys
import os
import argparse
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
from bcos.common import explanation_mode, gradient_to_image


# ═════════════════════════════════════════════════════════════════════════════
# 1. UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def apply_patch(x_rgb: Tensor, patch: Tensor, pos_y: int, pos_x: int) -> Tensor:
    """Adds a patch onto the 3-channel RGB image x_rgb."""
    x_pert = x_rgb.clone()
    _, _, hp, wp = patch.shape
    x_pert[..., pos_y:pos_y+hp, pos_x:pos_x+wp] = torch.clamp(x_pert[..., pos_y:pos_y+hp, pos_x:pos_x+wp] + patch, 0.0, 1.0)
    return x_pert

def to_bcos_input(x_rgb: Tensor) -> Tensor:
    """Converts 3-channel RGB [0,1] to 6-channel B-cos input (x, 1-x)."""
    return torch.cat([x_rgb, 1.0 - x_rgb], dim=1)

def extract_attribution(
    model: torch.nn.Module,
    x_6ch: Tensor,
    target_class: int,
) -> Dict[str, Any]:
    """Extract B-cos explanation using the built-in model.explain() method."""
    inp = x_6ch.detach().clone().requires_grad_(True)
    with explanation_mode(model):
        result = model.explain(inp, idx=target_class)
    return {
        "contribution_map": result["contribution_map"],
        "explanation": result["explanation"],
        "prediction": result["prediction"],
        "explained_class": result["explained_class_idx"],
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2. EVOLUTION STRATEGY OPTIMIZER
# ═════════════════════════════════════════════════════════════════════════════

class PatchEvolutionStrategyOptimizer:
    def __init__(
        self,
        param_shape: Tuple[int, ...],
        init_pos: Tuple[float, float],
        population_size: int = 50,
        elite_fraction: float = 0.2,
        sigma: float = 0.05,
        sigma_pos: float = 0.1,
        sigma_decay: float = 0.99,
        sigma_min: float = 1e-4,
        learning_rate: float = 1.0,
        lr_pos: float = 0.1,
        device: torch.device = torch.device('cpu')
    ):
        self.param_shape = param_shape
        self.population_size = population_size
        self.elite_size = max(1, int(population_size * elite_fraction))
        self.sigma = sigma
        self.sigma_pos = sigma_pos
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min
        self.lr = learning_rate
        self.lr_pos = lr_pos
        self.device = device
        
        # Initialize patch with zeros
        self.patch = torch.zeros(param_shape, device=device)
        self.pos = torch.tensor(init_pos, device=device, dtype=torch.float32)
        
    def ask(self) -> Tuple[Tensor, Tensor]:
        noise_patch = torch.randn(self.population_size, *self.param_shape, device=self.device)
        candidates_patch = self.patch.unsqueeze(0) + self.sigma * noise_patch
        # Clamp patch values to valid [-1, 1] additive range
        candidates_patch = candidates_patch.clamp(-1.0, 1.0)
        
        noise_pos = torch.randn(self.population_size, 2, device=self.device)
        candidates_pos = self.pos.unsqueeze(0) + self.sigma_pos * noise_pos
        candidates_pos = candidates_pos.clamp(0.0, 1.0)
        
        return candidates_patch, candidates_pos
        
    def tell(self, candidates_patch: Tensor, candidates_pos: Tensor, scores: Tensor) -> Tuple[Tensor, Tensor]:
        _, elite_idx = scores.topk(self.elite_size)
        
        # Update patch
        elite_patch = candidates_patch[elite_idx]
        elite_scores = scores[elite_idx]
        
        elite_scores = (elite_scores - elite_scores.mean()) / (elite_scores.std() + 1e-8)
        weights = F.softmax(elite_scores, dim=0)
        
        weights_patch = weights.clone()
        for _ in range(len(self.param_shape)):
            weights_patch = weights_patch.unsqueeze(-1)
            
        new_patch = (weights_patch * elite_patch).sum(dim=0)
        self.patch = (1 - self.lr) * self.patch + self.lr * new_patch
        self.patch = self.patch.clamp(-1.0, 1.0)
        
        # Update pos
        elite_pos = candidates_pos[elite_idx]
        weights_pos = weights.unsqueeze(-1)
        new_pos = (weights_pos * elite_pos).sum(dim=0)
        self.pos = (1 - self.lr_pos) * self.pos + self.lr_pos * new_pos
        self.pos = self.pos.clamp(0.0, 1.0)
        
        self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)
        self.sigma_pos = max(self.sigma_pos * self.sigma_decay, self.sigma_min)
        
        return self.patch, self.pos
        
    def get_params(self) -> Tuple[Tensor, Tensor]:
        return self.patch.clone(), self.pos.clone()


# ═════════════════════════════════════════════════════════════════════════════
# 3. SCORE COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_scores(
    model: torch.nn.Module, 
    x_rgb: Tensor, 
    candidates_patch: Tensor, 
    candidates_pos: Tensor, 
    H: int,
    W: int,
    target_class: int
) -> Tensor:
    """
    Evaluate multiple patch candidates at their respective positions.
    Goal: Force the model to predict a different class.
    We return the CW margin loss: max(other_logits) - logit[target_class].
    """
    scores = []
    N = candidates_patch.shape[0]
    batch_size = 100  # Prevent OOM
    
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch_patch = candidates_patch[i:i+batch_size]
            batch_pos = candidates_pos[i:i+batch_size]
            B = batch_patch.shape[0]
            
            x_pert = x_rgb.repeat(B, 1, 1, 1)
            _, hp, wp = batch_patch.shape[2:]
            
            for b in range(B):
                py_rel, px_rel = batch_pos[b].tolist()
                pos_y = int(py_rel * (H - hp))
                pos_x = int(px_rel * (W - wp))
                
                # Apply patch b to image b
                patch_b = batch_patch[b]
                x_pert[b:b+1, ..., pos_y:pos_y+hp, pos_x:pos_x+wp] = torch.clamp(
                    x_pert[b:b+1, ..., pos_y:pos_y+hp, pos_x:pos_x+wp] + patch_b, 0.0, 1.0
                )
            
            x_6ch = to_bcos_input(x_pert)
            outputs = model(x_6ch)
            
            # Score = max_other_logit - logit of target class (CW margin loss)
            outputs_clone = outputs.clone()
            outputs_clone[:, target_class] = -float('inf')
            max_other_logits = outputs_clone.max(dim=1)[0]
            
            cw_loss = max_other_logits - outputs[:, target_class]
            
            # L2 penalty
            l2_penalty = batch_patch.view(B, -1).pow(2).sum(dim=1)
            batch_scores = cw_loss - 0.05 * l2_penalty
            
            scores.append(batch_scores)
            # scores.append(-outputs[:, target_class])
            
    return torch.cat(scores)


# ═════════════════════════════════════════════════════════════════════════════
# 4. OPTIMIZATION LOOP
# ═════════════════════════════════════════════════════════════════════════════

def run_patch_attack(
    model: torch.nn.Module, 
    x_rgb: Tensor, 
    target_class: int,
    patch_size: int, 
    init_pos_y: int, 
    init_pos_x: int,
    generations: int = 50, 
    population_size: int = 50,
    elite_fraction: float = 0.2, 
    sigma: float = 0.05,
    sigma_pos: float = 0.1,
    sigma_decay: float = 0.99, 
    sigma_min: float = 1e-4,
    learning_rate: float = 1.0, 
    lr_pos: float = 0.1,
    verbose: bool = True
):
    device = x_rgb.device
    _, _, H, W = x_rgb.shape
    
    den_y = max(1, H - patch_size)
    den_x = max(1, W - patch_size)
    
    init_py_rel = min(1.0, init_pos_y / den_y)
    init_px_rel = min(1.0, init_pos_x / den_x)
    init_pos = (init_py_rel, init_px_rel)

    es = PatchEvolutionStrategyOptimizer(
        param_shape=(1, 3, patch_size, patch_size),
        init_pos=init_pos,
        population_size=population_size, elite_fraction=elite_fraction,
        sigma=sigma, sigma_pos=sigma_pos, sigma_decay=sigma_decay, sigma_min=sigma_min,
        learning_rate=learning_rate, lr_pos=lr_pos, device=device
    )
    
    history = []
    for gen in range(generations):
        candidates_patch, candidates_pos = es.ask()
        scores = compute_scores(model, x_rgb, candidates_patch, candidates_pos, H, W, target_class)
        es.tell(candidates_patch, candidates_pos, scores)
        
        best_score = scores.max().item()
        
        # Calculate real best logit
        best_idx = scores.argmax()
        best_patch = candidates_patch[best_idx]
        best_pos = candidates_pos[best_idx]
        
        py_rel, px_rel = best_pos.tolist()
        best_pos_y = int(py_rel * (H - patch_size))
        best_pos_x = int(px_rel * (W - patch_size))
        
        best_x_pert = apply_patch(x_rgb, best_patch, best_pos_y, best_pos_x)
        with torch.no_grad():
            best_out = model(to_bcos_input(best_x_pert))
            best_logit = best_out[0, target_class].item()
            
        history.append((gen, best_score, best_logit, es.sigma))
        
        if verbose:
            print(f"  Gen {gen+1:3d}/{generations} | Target Logit: {best_logit:7.3f} | Score: {best_score:7.3f} | σ: {es.sigma:.4f}")
            
    final_patch, final_pos = es.get_params()
    final_py_rel, final_px_rel = final_pos.tolist()
    final_pos_y = int(final_py_rel * (H - patch_size))
    final_pos_x = int(final_px_rel * (W - patch_size))
    
    return final_patch, final_pos_y, final_pos_x, history


# ═════════════════════════════════════════════════════════════════════════════
# 5. VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

def visualize_patch_results(
    x_orig_rgb, x_pert_rgb, patch,
    attr_orig, attr_pert, target_class, logit_orig,
    pred_class_pert, logit_pert_tgt, logit_pert_pred,
    history, save_path
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(20, 11), facecolor="#0d1117")
    fig.suptitle(
        "B-cos Counterfactual Patch Attack",
        fontsize=22, fontweight="bold", color="white", y=0.96,
    )
    
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.25,
                  top=0.88, bottom=0.08, left=0.04, right=0.96)
                  
    label_color = "#c9d1d9"

    def style_ax(ax, title):
        ax.set_title(title, fontsize=12, fontweight="bold", color=label_color, pad=10)
        ax.set_facecolor("#161b22")
        ax.tick_params(colors=label_color, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#30363d")
            
    # Row 1: Images & Patch
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(x_orig_rgb.squeeze(0).permute(1, 2, 0).cpu().numpy())
    ax1.set_xticks([]); ax1.set_yticks([])
    style_ax(ax1, f"Original Image\nPred Class: {target_class} (logit: {logit_orig:.2f})")
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(x_pert_rgb.squeeze(0).permute(1, 2, 0).cpu().numpy())
    ax2.set_xticks([]); ax2.set_yticks([])
    style_ax(ax2, f"Attacked Image\nNew Pred: {pred_class_pert} (logit: {logit_pert_pred:.2f})\nTgt Class Logit: {logit_pert_tgt:.2f}")
    
    ax3 = fig.add_subplot(gs[0, 2])
    # Show the patch scaled up (shifted to [0, 1] for visualization)
    patch_vis = (patch.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0
    ax3.imshow(patch_vis)
    ax3.set_xticks([]); ax3.set_yticks([])
    style_ax(ax3, "Adversarial Patch (zoom)")
    
    # Plot convergence
    ax4 = fig.add_subplot(gs[0, 3])
    gens = [h[0]+1 for h in history]
    logits = [h[2] for h in history]
    ax4.plot(gens, logits, color="#f0883e", linewidth=2.5)
    ax4.set_xlabel("Generation", color=label_color)
    ax4.set_ylabel("Target Class Logit", color=label_color)
    style_ax(ax4, "Target Class Logit Decay")
    ax4.grid(True, alpha=0.15, color="white")
    
    # Row 2: Explanations
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(attr_orig["explanation"])
    ax5.set_xticks([]); ax5.set_yticks([])
    style_ax(ax5, f"Original Explanation\n(w.r.t Target Class {target_class})")
    
    ax6 = fig.add_subplot(gs[1, 1])
    bcos.plot_contribution_map(attr_orig["contribution_map"].squeeze(0), ax=ax6)
    style_ax(ax6, "Original Contribution Map")

    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(attr_pert["explanation"])
    ax7.set_xticks([]); ax7.set_yticks([])
    style_ax(ax7, f"Perturbed Explanation\n(w.r.t Target Class {target_class})")
    
    ax8 = fig.add_subplot(gs[1, 3])
    bcos.plot_contribution_map(attr_pert["contribution_map"].squeeze(0), ax=ax8)
    style_ax(ax8, "Perturbed Contribution Map")
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"\n  ✅ Results plot saved to: {save_path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# 6. DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_rgb_image(image_path: Optional[str] = None, device: torch.device = torch.device('cpu')) -> Tuple[Tensor, str]:
    from PIL import Image
    from torchvision import transforms as T
    import random
    
    if image_path is None:
        imagenet_root = "/kaggle/input/imagenet1kvalid"
        if os.path.isdir(imagenet_root):
            class_dirs = sorted([d for d in os.listdir(imagenet_root) if os.path.isdir(os.path.join(imagenet_root, d))])
            rng = random.Random(42)
            chosen_class = rng.choice(class_dirs)
            class_path = os.path.join(imagenet_root, chosen_class)
            imgs = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]
            chosen_img = rng.choice(imgs)
            image_path = os.path.join(class_path, chosen_img)
            print(f"  📸  Using ImageNet image: class={chosen_class}, file={chosen_img}")
        else:
            raise FileNotFoundError("ImageNet not found and no image provided.")
            
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor()
    ])
    x = transform(img)
    return x.unsqueeze(0).to(device), image_path


# ═════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    import random
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser(description="B-cos Counterfactual Patch Attack using ES")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--model", type=str, default="resnet18", help="B-cos model (default: resnet18)")
    parser.add_argument("--patch-size", type=int, default=40, help="Size of the square patch (default: 40)")
    parser.add_argument("--pos-y", type=int, default=-1, help="Patch Y pos (default: -1 for center)")
    parser.add_argument("--pos-x", type=int, default=-1, help="Patch X pos (default: -1 for center)")
    
    parser.add_argument("--generations", type=int, default=500, help="Number of ES generations (default: 50)")
    parser.add_argument("--population", type=int, default=100, help="ES population size (default: 50)")
    parser.add_argument("--elite-fraction", type=float, default=0.2, help="Fraction of elite candidates (default: 0.2)")
    parser.add_argument("--sigma", type=float, default=0.05, help="Initial noise std dev (default: 0.05)")
    parser.add_argument("--sigma-decay", type=float, default=0.99, help="Sigma decay per generation (default: 0.99)")
    parser.add_argument("--lr", type=float, default=1.0, help="ES learning rate (default: 1.0)")
    
    parser.add_argument("--device", type=str, default="auto", help="Device: cpu, cuda, auto")
    parser.add_argument("--save", type=str, default="patch_attack_results.png", help="Path to save result figure")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 72)
    print("  B-cos Counterfactual Patch Attack (ES)")
    print("=" * 72)
    print(f"  Device       : {device}")
    print(f"  Model        : {args.model}")
    print(f"  Patch size   : {args.patch_size}x{args.patch_size}")
    
    # ── 1. Load model ──
    print("\n▸ Loading B-cos model …")
    model_fn = getattr(bcos.pretrained, args.model)
    model = model_fn(pretrained=True).to(device).eval()
    
    # ── 2. Load Image ──
    print("\n▸ Loading original image …")
    x_rgb, image_path = load_rgb_image(args.image, device=device)
    
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = image_name
    os.makedirs(out_dir, exist_ok=True)
    
    import torchvision.utils as vutils
    vutils.save_image(x_rgb, os.path.join(out_dir, "before.png"))
    print(f"  ✓ Saved original image to {os.path.join(out_dir, 'before.png')}")
    
    # Determine patch position
    if args.pos_y == -1 or args.pos_x == -1:
        H, W = x_rgb.shape[2], x_rgb.shape[3]
        pos_y = (H - args.patch_size) // 2
        pos_x = (W - args.patch_size) // 2
    else:
        pos_y, pos_x = args.pos_y, args.pos_x
        
    print(f"  ✓ Patch position (Y, X) : ({pos_y}, {pos_x})")

    # ── 3. Evaluate Original ──
    x_6ch_orig = to_bcos_input(x_rgb)
    with torch.no_grad():
        out_orig = model(x_6ch_orig)
        pred_class_orig = out_orig.argmax(1).item()
        logit_orig = out_orig[0, pred_class_orig].item()
    target_class = pred_class_orig
    print(f"  ✓ Original Prediction   : Class {target_class} (logit: {logit_orig:.3f})")
    
    print(f"  ✓ Top 5 Original Predictions:")
    top5_logits_orig, top5_classes_orig = out_orig[0].topk(5)
    for i in range(5):
        print(f"      {i+1}. Class {top5_classes_orig[i].item():4d} | Logit: {top5_logits_orig[i].item():7.3f}")
    
    # Extract original attribution
    attr_orig = extract_attribution(model, x_6ch_orig, target_class=target_class)

    # ── 4. Run Optimization ──
    print("\n▸ Running ES Patch Optimization (Target: minimize original logit) …")
    final_patch, final_pos_y, final_pos_x, history = run_patch_attack(
        model, x_rgb, target_class, 
        patch_size=args.patch_size, init_pos_y=pos_y, init_pos_x=pos_x,
        generations=args.generations, population_size=args.population,
        elite_fraction=args.elite_fraction, sigma=args.sigma,
        sigma_decay=args.sigma_decay, learning_rate=args.lr,
        verbose=True
    )
    print(f"  ✓ Final Patch position (Y, X) : ({final_pos_y}, {final_pos_x})")

    # ── 5. Evaluate Perturbed ──
    final_patch = final_patch.clamp(-0.3, 0.3)
    x_pert_rgb = apply_patch(x_rgb, final_patch, final_pos_y, final_pos_x)
    
    vutils.save_image(x_pert_rgb, os.path.join(out_dir, "after.png"))
    print(f"  ✓ Saved perturbed image to {os.path.join(out_dir, 'after.png')}")
    
    x_6ch_pert = to_bcos_input(x_pert_rgb)
    with torch.no_grad():
        out_pert = model(x_6ch_pert)
        pred_class_pert = out_pert.argmax(1).item()
        logit_pert_tgt = out_pert[0, target_class].item()
        logit_pert_pred = out_pert[0, pred_class_pert].item()
        
    print(f"\n▸ Attack Complete")
    print(f"  ✓ Target Class Logit : {logit_orig:.3f} -> {logit_pert_tgt:.3f}")
    
    print(f"  ✓ Top 5 Perturbed Predictions:")
    top5_logits, top5_classes = out_pert[0].topk(5)
    for i in range(5):
        print(f"      {i+1}. Class {top5_classes[i].item():4d} | Logit: {top5_logits[i].item():7.3f}")

    if pred_class_pert != target_class:
        print(f"  ✓ Class Falsified! New Pred: Class {pred_class_pert} (logit: {logit_pert_pred:.3f})")
    else:
        print(f"  ✓ Class Unchanged. Failed to cross decision boundary.")

    # Extract perturbed attribution (still explaining the target class to show disruption)
    print("\n▸ Extracting explanation for perturbed image …")
    attr_pert = extract_attribution(model, x_6ch_pert, target_class=target_class)

    # ── 6. Visualise ──
    print("\n▸ Generating visualization …")
    
    save_plot_path = args.save if args.save != "patch_attack_results.png" else os.path.join(out_dir, "patch_attack_results.png")
    
    visualize_patch_results(
        x_orig_rgb=x_rgb,
        x_pert_rgb=x_pert_rgb,
        patch=final_patch,
        attr_orig=attr_orig,
        attr_pert=attr_pert,
        target_class=target_class,
        logit_orig=logit_orig,
        pred_class_pert=pred_class_pert,
        logit_pert_tgt=logit_pert_tgt,
        logit_pert_pred=logit_pert_pred,
        history=history,
        save_path=save_plot_path
    )
    
    print("\n" + "=" * 72)
    print("  Done!")
    print("=" * 72)

if __name__ == "__main__":
    main()