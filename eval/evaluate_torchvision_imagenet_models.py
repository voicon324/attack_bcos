"""
Evaluate torchvision ImageNet models on an image CSV.

The script uses each model's default pretrained weights with a temporary,
shared image transform so every model is evaluated on the same 224x224
center-cropped tensor input.
"""

from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tv_models
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT
DEFAULT_CSV_PATH = SCRIPT_DIR / "data" / "used_images_500.csv"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "artifacts" / "outputs" / "result" / "original_model_eval_summary.csv"
IMAGENET_SL = 224

# Keep these aliases aligned with the attack scripts so model names mean the
# same thing across the repo.
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


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


DEFAULT_MODEL_ORDER = unique_preserve_order(TORCHVISION_MODEL_MAP.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate original torchvision ImageNet models on a CSV of images.",
    )
    parser.add_argument(
        "--images-csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="CSV containing an image_path column.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=(
            "Model names to evaluate. Use 'all' for every unique torchvision model "
            f"mapped in this repo: {', '.join(DEFAULT_MODEL_ORDER)}"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Evaluation batch size per model.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--target-correct",
        type=int,
        default=404,
        help="Highlight models with exactly this many correct predictions.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Where to save the summary table.",
    )
    parser.add_argument(
        "--fast-runtime",
        action="store_true",
        help="Enable faster CUDA settings instead of stricter reproducibility settings.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress for every batch.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def configure_runtime(device: torch.device, fast_runtime: bool) -> None:
    if device.type != "cuda":
        return
    if fast_runtime:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cudnn, "deterministic"):
        torch.backends.cudnn.deterministic = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def load_image_paths_from_csv(csv_path: Path) -> List[str]:
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if "image_path" not in (reader.fieldnames or []):
            raise ValueError(f"CSV {csv_path} must contain an 'image_path' column.")
        image_paths = [(row.get("image_path") or "").strip() for row in reader]
    image_paths = [path for path in image_paths if path]
    if not image_paths:
        raise ValueError(f"CSV {csv_path} does not contain any valid image_path values.")
    return image_paths


def infer_ground_truth_from_path(image_path: str) -> int:
    parent_name = Path(image_path).parent.name.strip()
    if not parent_name.isdigit():
        raise ValueError(
            f"Cannot infer ground truth from image path {image_path!r}; "
            "expected parent directory to be numeric."
        )
    return int(parent_name)


def resolve_model_names(requested: Sequence[str]) -> List[str]:
    normalized = [name.strip() for name in requested if name.strip()]
    if not normalized:
        raise ValueError("No model names were provided.")
    if any(name == "all" for name in normalized):
        return DEFAULT_MODEL_ORDER

    resolved: List[str] = []
    for name in normalized:
        resolved.append(TORCHVISION_MODEL_MAP.get(name, name))
    return unique_preserve_order(resolved)


def describe_weight_enum(weights: object) -> str:
    enum_name = getattr(weights.__class__, "__name__", type(weights).__name__)
    member_name = getattr(weights, "name", None)
    if member_name is None:
        return str(weights)
    return f"{enum_name}.{member_name}"


def resolve_model_weights(model_name: str):
    weights_enum = tv_models.get_model_weights(model_name)
    weight_v1 = getattr(weights_enum, "IMAGENET1K_V1", None)
    if weight_v1 is not None:
        return weight_v1
    fallback = weights_enum.DEFAULT
    print(
        f"  warning      : {model_name} does not provide IMAGENET1K_V1; "
        f"falling back to {describe_weight_enum(fallback)}"
    )
    return fallback


class CsvImageDataset(Dataset):
    def __init__(self, image_paths: Sequence[str], transform) -> None:
        self.samples: List[Tuple[str, int]] = [
            (image_path, infer_ground_truth_from_path(image_path))
            for image_path in image_paths
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_path, target = self.samples[index]
        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
        return self.transform(image_rgb), target


def build_loader(
    image_paths: Sequence[str],
    transform,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = CsvImageDataset(image_paths, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def evaluate_model(
    model_name: str,
    image_paths: Sequence[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    verbose: bool,
) -> Dict[str, object]:
    weights = resolve_model_weights(model_name)
    preprocess = transforms.Compose(
        [
            transforms.Resize(IMAGENET_SL),
            transforms.CenterCrop(IMAGENET_SL),
            transforms.ToTensor(),
            transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
        ]
    )
    resize_size = IMAGENET_SL
    crop_size = IMAGENET_SL

    loader = build_loader(
        image_paths=image_paths,
        transform=preprocess,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = tv_models.get_model(model_name, weights=weights).to(device).eval()

    correct = 0
    total = 0
    start_time = time.perf_counter()
    with torch.inference_mode():
        for batch_idx, (inputs, targets) in enumerate(loader, start=1):
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += int((preds == targets).sum().item())
            total += int(targets.numel())
            if verbose:
                print(
                    f"    batch {batch_idx:03d}: running accuracy "
                    f"{correct}/{total} ({correct / total * 100.0:.2f}%)"
                )

    elapsed_sec = time.perf_counter() - start_time
    del loader
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "weights": describe_weight_enum(weights),
        "resize_size": resize_size,
        "crop_size": crop_size,
        "correct": correct,
        "total": total,
        "accuracy_pct": correct / total * 100.0,
        "elapsed_sec": elapsed_sec,
    }


def save_results_csv(output_csv: Path, rows: Sequence[Dict[str, object]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "weights",
        "resize_size",
        "crop_size",
        "correct",
        "total",
        "accuracy_pct",
        "elapsed_sec",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0.")

    device = resolve_device(args.device)
    configure_runtime(device, fast_runtime=args.fast_runtime)

    image_paths = load_image_paths_from_csv(args.images_csv)
    model_names = resolve_model_names(args.models)

    print("=" * 72)
    print("  Original Model Evaluation")
    print("=" * 72)
    print(f"  CSV          : {args.images_csv}")
    print(f"  Images       : {len(image_paths)}")
    print(f"  Models       : {', '.join(model_names)}")
    print(f"  Device       : {device}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Workers      : {args.num_workers}")
    print(f"  Fast runtime : {args.fast_runtime}")
    print(f"  Target count : {args.target_correct}")
    print()

    results: List[Dict[str, object]] = []
    for idx, model_name in enumerate(model_names, start=1):
        print("-" * 72)
        print(f"[{idx}/{len(model_names)}] {model_name}")
        print("-" * 72)
        result = evaluate_model(
            model_name=model_name,
            image_paths=image_paths,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            verbose=args.verbose,
        )
        results.append(result)
        print(
            f"  weights      : {result['weights']}\n"
            f"  resize/crop  : {result['resize_size']} / {result['crop_size']}\n"
            f"  correct      : {result['correct']}/{result['total']} "
            f"({result['accuracy_pct']:.2f}%)\n"
            f"  elapsed      : {result['elapsed_sec']:.1f}s"
        )

    results_sorted = sorted(results, key=lambda row: (-int(row["correct"]), str(row["model"])))
    save_results_csv(args.output_csv, results_sorted)

    matches = [row for row in results_sorted if int(row["correct"]) == args.target_correct]

    print("\n" + "=" * 72)
    print("  Summary")
    print("=" * 72)
    for row in results_sorted:
        marker = " <==" if int(row["correct"]) == args.target_correct else ""
        print(
            f"  {str(row['model']):<18} {int(row['correct']):>3}/{int(row['total'])} "
            f"({float(row['accuracy_pct']):6.2f}%){marker}"
        )

    print(f"\nSaved summary to {args.output_csv}")
    if matches:
        matched_names = ", ".join(str(row["model"]) for row in matches)
        print(f"Models hitting {args.target_correct}/{len(image_paths)}: {matched_names}")
    else:
        print(f"No model hit exactly {args.target_correct}/{len(image_paths)}.")


if __name__ == "__main__":
    main()
