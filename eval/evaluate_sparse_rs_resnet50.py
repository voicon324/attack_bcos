"""
Evaluate the original sparse-rs torchvision ResNet-50 on an image CSV.

This matches the non-attack model path from fra31/sparse-rs:
* torchvision ResNet-50 with ImageNet V1 weights
* Resize(224) -> CenterCrop(224) -> ToTensor()
* ImageNet normalization applied inside the model wrapper
"""

from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tv_models
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT
DEFAULT_CSV_PATH = SCRIPT_DIR / "data" / "used_images_500.csv"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "artifacts" / "outputs" / "result" / "sparse_rs_resnet50_eval_summary.csv"
IMAGENET_SL = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the original sparse-rs ResNet-50 on a CSV of images.",
    )
    parser.add_argument(
        "--images-csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="CSV containing an image_path column.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Evaluation batch size.",
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
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Where to save the summary row.",
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


class SparseRsPretrainedModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mu", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("sigma", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mu) / self.sigma)


def build_sparse_rs_model(device: torch.device) -> Tuple[nn.Module, str]:
    model_label = "torchvision.models.resnet50(pretrained=True)"

    if hasattr(tv_models, "ResNet50_Weights"):
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1
        model = tv_models.resnet50(weights=weights)
        model_label = "torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)"
    else:
        model = tv_models.resnet50(pretrained=True)

    wrapped = SparseRsPretrainedModel(model).to(device).eval()
    return wrapped, model_label


def build_loader(
    image_paths: Sequence[str],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGENET_SL),
            transforms.CenterCrop(IMAGENET_SL),
            transforms.ToTensor(),
        ]
    )
    dataset = CsvImageDataset(image_paths, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def evaluate_sparse_rs_resnet50(
    image_paths: Sequence[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    verbose: bool,
) -> Dict[str, object]:
    model, model_impl = build_sparse_rs_model(device)
    loader = build_loader(
        image_paths=image_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

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
        "model": "sparse-rs pt_resnet",
        "model_impl": model_impl,
        "resize_size": IMAGENET_SL,
        "crop_size": IMAGENET_SL,
        "normalization": "in_model_wrapper",
        "correct": correct,
        "total": total,
        "accuracy_pct": correct / total * 100.0,
        "elapsed_sec": elapsed_sec,
    }


def save_results_csv(output_csv: Path, row: Dict[str, object]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "model_impl",
        "resize_size",
        "crop_size",
        "normalization",
        "correct",
        "total",
        "accuracy_pct",
        "elapsed_sec",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0.")

    device = resolve_device(args.device)
    configure_runtime(device, fast_runtime=args.fast_runtime)

    image_paths = load_image_paths_from_csv(args.images_csv)

    print("=" * 72)
    print("  sparse-rs ResNet-50 Evaluation")
    print("=" * 72)
    print(f"  CSV            : {args.images_csv}")
    print(f"  Images         : {len(image_paths)}")
    print("  Model          : pt_resnet")
    print("  Transform      : Resize(224) -> CenterCrop(224) -> ToTensor()")
    print("  Normalize      : inside model wrapper")
    print(f"  Device         : {device}")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  Workers        : {args.num_workers}")
    print(f"  Fast runtime   : {args.fast_runtime}")
    print()

    result = evaluate_sparse_rs_resnet50(
        image_paths=image_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        verbose=args.verbose,
    )
    save_results_csv(args.output_csv, result)

    print("-" * 72)
    print(f"  model_impl     : {result['model_impl']}")
    print(f"  correct        : {result['correct']}/{result['total']}")
    print(f"  accuracy       : {result['accuracy_pct']:.2f}%")
    print(f"  elapsed        : {result['elapsed_sec']:.1f}s")
    print(f"\nSaved summary to {args.output_csv}")


if __name__ == "__main__":
    main()
