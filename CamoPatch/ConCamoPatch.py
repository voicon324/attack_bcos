from torchvision import transforms
from ImageNetModels import ImageNetModel
from LossFunctions import UnTargeted
import numpy as np
import argparse
import sys
import torch
from pathlib import Path
from PIL import Image


def pytorch_switch(tensor_image):
    return tensor_image.permute(1, 2, 0)


def parse_linf(value):
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)


def describe_attack_model(model_idx, model_source):
    model_names = {
        ("bcos", 1): "bcos_resnet50",
        ("torchvision", 0): "torchvision_vgg16_bn",
        ("torchvision", 1): "torchvision_resnet50",
    }
    return model_names.get((model_source, model_idx), f"{model_source}_model_{model_idx}")


def resolve_bcos_guide_model(model, model_idx, device):
    if getattr(model, "model_source", None) == "bcos":
        return model
    return ImageNetModel(model_idx, device=device, model_source="bcos")


def resolve_fixed_bcos_position(model, x_rgb_chw, true_label, patch_size, model_idx, device, guide=None):
    repo_root = Path(__file__).resolve().parents[1]
    attacks_dir = repo_root / "attacks"
    for path in (repo_root, attacks_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    from attacks.explain_guided_circle_rgb_es_patch import (
        extract_attribution,
        find_best_patch_position_from_contribution_margin,
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
        # CamoPatch's untargeted loss is anchored on true_label, so use the
        # same class for the B-cos contribution-margin position.
        target_classes = torch.tensor([int(true_label)], device=outputs.device, dtype=torch.long)
        secondary_class = int(resolve_runner_up_classes(outputs, target_classes)[0].item())
        guide_prediction = int(outputs.argmax(dim=1)[0].item())

    primary_guidance = extract_attribution(guide_model, x_bcos, target_class=int(true_label))
    secondary_guidance = extract_attribution(guide_model, x_bcos, target_class=secondary_class)
    pos_y, pos_x = find_best_patch_position_from_contribution_margin(
        primary_guidance["contribution_map"],
        secondary_guidance["contribution_map"],
        patch_size,
    )
    return np.array([pos_y, pos_x], dtype=np.int64), guide_prediction, secondary_class


from CamoPatch import Attack

IMAGENET_SL = 224


if __name__ == "__main__":

    # Match evaluate_original_models_on_csv.py spatial preprocessing. The tensor
    # stays in [0,1] because ImageNetModel.predict normalizes torchvision inputs.
    load_image = transforms.Compose([
        transforms.Resize(IMAGENET_SL),
        transforms.CenterCrop(IMAGENET_SL),
        transforms.ToTensor()
    ])

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="0 or 1. With --model_source auto, 1 uses B-cos ResNet-50.", type=int, default=1)
    parser.add_argument(
        "--model_source",
        choices=("auto", "bcos", "torchvision"),
        default="auto",
        help="auto uses local/uploaded B-cos weights for ResNet-50; torchvision keeps the original CamoPatch models.",
    )
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--temp", type=float, default=300.)
    parser.add_argument("--mut", type=float, default=0.3)
    parser.add_argument("--s", type=int, default=16)
    parser.add_argument("--queries", type=int, default=10000)
    parser.add_argument("--li", type=int, default=4)
    parser.add_argument(
        "--linf",
        type=parse_linf,
        default=None,
        help="Optional L-infinity bound in normalized [0,1] units. Fractions like 8/255 are accepted.",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda, cuda:0, or cpu. Default: cpu, matching the original code path")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="1 keeps the original sequential attack. >1 evaluates parallel patch candidates per GPU batch.",
    )
    parser.add_argument(
        "--parallel_locations",
        action="store_true",
        help="Also batch location candidates. Faster, but changes the location-update schedule more than batching patch candidates.",
    )
    parser.add_argument(
        "--fixed_bcos_position",
        action="store_true",
        help=(
            "Choose the initial patch position from the B-cos contribution-margin rule "
            "used by attacks/explain_guided_circle_rgb_es_patch.py, then keep that "
            "position fixed for all queries."
        ),
    )
    parser.add_argument(
        "--init_bcos_position",
        action="store_true",
        help=(
            "Choose the initial patch position from the B-cos contribution-margin rule "
            "used by attacks/explain_guided_circle_rgb_es_patch.py, then still allow "
            "normal CamoPatch location updates."
        ),
    )
    parser.add_argument(
        "--trace_every",
        type=int,
        default=1,
        help="Save genotype/location trace every N queries. Use 0 to disable trace saving for faster runs and smaller .npy files.",
    )
    parser.add_argument(
        "--no_save_images",
        action="store_false",
        dest="save_images",
        help="Only save the .npy result and skip PNG exports.",
    )
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--image_dir", type=str, help="Image File directory")
    parser.add_argument("--true_label", type=int, help="Number of the correct label of ImageNet inputted image")
    parser.add_argument("--save_directory", type=str, help="Where to store the .npy files with the results")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    model = ImageNetModel(args.model, device=args.device, model_source=args.model_source)
    attack_model_source = model.model_source
    attack_model_name = describe_attack_model(args.model, attack_model_source)
    print(f"Attacking model: {attack_model_name}")

    image_dir = args.image_dir
    x_test = load_image(Image.open(image_dir).convert("RGB"))
    initial_loc = None
    location_source = None
    if args.fixed_bcos_position or args.init_bcos_position:
        initial_loc, guide_prediction, secondary_class = resolve_fixed_bcos_position(
            model,
            x_test,
            args.true_label,
            args.s,
            args.model,
            args.device,
        )
        location_source = "bcos_contribution_margin_fixed" if args.fixed_bcos_position else "bcos_contribution_margin_init"
        movement = "disabled" if args.fixed_bcos_position else "enabled"
        print(
            "Using B-cos initial position "
            f"(y={int(initial_loc[0])}, x={int(initial_loc[1])}); "
            f"location_updates={movement}; "
            f"guide_pred={guide_prediction}, secondary={secondary_class}"
        )

    loss = UnTargeted(model, args.true_label, to_pytorch=True)
    x = pytorch_switch(x_test).detach().numpy()
    params = {
        "x": x,
        "eps": args.s**2,
        "n_queries": args.queries,
        "save_directory": args.save_directory + ".npy",
        "c": x.shape[2],
        "h": x.shape[0],
        "w": x.shape[1],
        "N": args.N,
        "update_loc_period": args.li,
        "mut": args.mut,
        "temp": args.temp,
        "eps_linf": args.linf,
        "attack_model": attack_model_name,
        "attack_model_source": attack_model_source,
        "attack_model_index": args.model,
        "eval_batch_size": args.batch_size,
        "parallel_locations": args.parallel_locations,
        "fixed_location": args.fixed_bcos_position,
        "initial_loc": initial_loc,
        "location_source": location_source,
        "trace_every": args.trace_every,
        "save_images": args.save_images,
    }
    attack = Attack(params)
    attack.optimise(loss)
