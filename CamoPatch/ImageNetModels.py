import os
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import torch
from torchvision import models as torch_models
from urllib.parse import urlparse

DEFAULT_DEVICE = "cpu"

LEGACY_TORCHVISION_MODEL_SPECS = [
    ("vgg16_bn", torch_models.vgg16_bn, getattr(torch_models, "VGG16_BN_Weights", None)),
    ("resnet50", torch_models.resnet50, getattr(torch_models, "ResNet50_Weights", None)),
]

BCOS_MODEL_ALIASES = {
    "1": "resnet50",
    "resnet_50": "resnet50",
    "resnet50": "resnet50",
    "resnet_18": "resnet18",
    "resnet18": "resnet18",
    "densenet_121": "densenet121",
    "densenet121": "densenet121",
    "convnext_tiny": "convnext_tiny",
    "convnext_base": "convnext_base",
    "vitc_s": "vitc_s_patch1_14",
    "vitc_s_patch1_14": "vitc_s_patch1_14",
    "vitc_b": "vitc_b_patch1_14",
    "vitc_b_patch1_14": "vitc_b_patch1_14",
}

TORCHVISION_MODEL_ALIASES = {
    "0": "vgg16_bn",
    "1": "resnet50",
    "vgg16_bn": "vgg16_bn",
    "resnet50": "resnet50",
    "resnet18": "resnet18",
    "densenet121": "densenet121",
    "convnext_tiny": "convnext_tiny",
    "convnext_base": "convnext_base",
}


def _normalize_model_key(model):
    key = str(model).strip().lower().replace("-", "_")
    if not key:
        raise ValueError("Model id/name must not be empty.")
    return key


def canonical_bcos_model_name(model):
    key = _normalize_model_key(model)
    try:
        return BCOS_MODEL_ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(BCOS_MODEL_ALIASES))
        raise ValueError(f"Unsupported B-cos model '{model}'. Supported: {supported}") from exc


def canonical_torchvision_model_name(model):
    key = _normalize_model_key(model)
    return TORCHVISION_MODEL_ALIASES.get(key, key)


def _resolve_device(device):
    device = device or DEFAULT_DEVICE
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return torch.device(device)


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _torch_load_bytes(data):
    try:
        return torch.load(BytesIO(data), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(BytesIO(data), map_location="cpu")


def _load_from_zip(zip_path, filename, folder_name):
    with zipfile.ZipFile(zip_path) as archive:
        preferred = [
            filename,
            f"{folder_name}/{filename}",
            f"weights/{folder_name}/{filename}",
        ]
        names = set(archive.namelist())
        for name in preferred:
            if name in names:
                return _torch_load_bytes(archive.read(name))

        matches = [name for name in archive.namelist() if Path(name).name == filename]
        if matches:
            return _torch_load_bytes(archive.read(matches[0]))
    return None


def _yield_torchvision_root_candidates(root, filename):
    yield root / filename
    yield root / "weights" / filename
    yield root / "weights" / "torchvision-imagenet" / filename
    yield root / "weights" / "original-imagenet" / filename
    yield root / "torchvision-imagenet" / filename
    yield root / "original-imagenet" / filename
    yield root / "torchvision-imagenet.zip"
    yield root / "weights" / "torchvision-imagenet.zip"


def _iter_torchvision_weight_candidates(filename):
    env_file = os.environ.get("TORCHVISION_WEIGHT_FILE")
    if env_file:
        yield Path(env_file)

    for env_name in ("TORCHVISION_WEIGHTS_DIR", "MODEL_WEIGHTS_DIR", "BCOS_WEIGHTS_DIR", "WEIGHTS_DIR"):
        env_dir = os.environ.get(env_name)
        if env_dir:
            yield from _yield_torchvision_root_candidates(Path(env_dir), filename)

    repo_root = Path(__file__).resolve().parents[1]
    yield from _yield_torchvision_root_candidates(repo_root, filename)

    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir():
        yield from _yield_torchvision_root_candidates(kaggle_working, filename)

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for dataset_dir in sorted(p for p in kaggle_input.iterdir() if p.is_dir()):
            yield from _yield_torchvision_root_candidates(dataset_dir, filename)


def _load_local_torchvision_state_dict(weights):
    url = getattr(weights, "url", None)
    if not url:
        return None

    filename = Path(urlparse(url).path).name
    for candidate in _iter_torchvision_weight_candidates(filename):
        if candidate.is_file():
            if candidate.suffix.lower() == ".zip":
                state_dict = _load_from_zip(candidate, filename, "torchvision-imagenet")
                if state_dict is None:
                    continue
                print(f"Loading local torchvision weights from {candidate}:{filename}")
                return state_dict

            print(f"Loading local torchvision weights from {candidate}")
            return _torch_load(candidate)
    return None


def _format_torchvision_search_paths(weights):
    url = getattr(weights, "url", None)
    if not url:
        return ""
    filename = Path(urlparse(url).path).name
    return "\n".join(f"  - {candidate}" for candidate in _iter_torchvision_weight_candidates(filename))


def _load_torchvision_model(model):
    key = _normalize_model_key(model)
    if key.isdigit() and int(key) < len(LEGACY_TORCHVISION_MODEL_SPECS):
        model_name, factory, weights_enum = LEGACY_TORCHVISION_MODEL_SPECS[int(key)]
    else:
        model_name = canonical_torchvision_model_name(model)
        try:
            weights_enum = torch_models.get_model_weights(model_name)
            factory = lambda weights=None: torch_models.get_model(model_name, weights=weights)
        except Exception as exc:
            raise ValueError(f"Unsupported torchvision model '{model}'.") from exc

    if weights_enum is not None:
        weights = getattr(weights_enum, "IMAGENET1K_V1", weights_enum.DEFAULT)
        state_dict = _load_local_torchvision_state_dict(weights)
        if state_dict is not None:
            model = factory(weights=None)
            model.load_state_dict(state_dict)
            return model
        if os.environ.get("ALLOW_WEIGHT_DOWNLOAD") == "1":
            return factory(weights=weights)
        raise RuntimeError(
            "Could not load torchvision pretrained weights offline.\n"
            f"Expected local file for {weights} in one of:\n{_format_torchvision_search_paths(weights)}\n"
            "Attach the hkhnhduy/weights-bcos Kaggle dataset and/or set MODEL_WEIGHTS_DIR=/kaggle/input/weights-bcos.\n"
            "Set ALLOW_WEIGHT_DOWNLOAD=1 only if network downloads are intended."
        )
    return factory(pretrained=True)


def _ensure_bcos_importable():
    repo_root = Path(__file__).resolve().parents[1]
    bcos_repo = repo_root / "B-cos-v2"
    if bcos_repo.is_dir() and str(bcos_repo) not in sys.path:
        sys.path.insert(0, str(bcos_repo))

    local_weights_dir = repo_root / "weights" / "bcos-imagenet"
    if local_weights_dir.is_dir() and not os.environ.get("BCOS_WEIGHTS_DIR"):
        os.environ["BCOS_WEIGHTS_DIR"] = str(local_weights_dir)

    import bcos
    return bcos


def _load_bcos_model(model):
    bcos = _ensure_bcos_importable()
    model_name = canonical_bcos_model_name(model)
    model_fn = getattr(bcos.pretrained, model_name)
    return model_fn(pretrained=True)


def _load_pretrained_model(model, model_source):
    model_source = model_source.lower()
    if model_source == "auto":
        key = _normalize_model_key(model)
        if key in BCOS_MODEL_ALIASES:
            return _load_bcos_model(model), "bcos", canonical_bcos_model_name(model)
        model_name = canonical_torchvision_model_name(model)
        return _load_torchvision_model(model), "torchvision", model_name
    if model_source == "bcos":
        model_name = canonical_bcos_model_name(model)
        return _load_bcos_model(model), "bcos", model_name
    if model_source == "torchvision":
        model_name = canonical_torchvision_model_name(model)
        return _load_torchvision_model(model), "torchvision", model_name
    raise ValueError("model_source must be one of: auto, bcos, torchvision")


def _as_batched_chw(x, device):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
    else:
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)

    if x.ndim == 3:
        if x.shape[0] == 3:
            x = x.unsqueeze(0)
        elif x.shape[-1] == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(f"Expected CHW or HWC image, got shape {tuple(x.shape)}")
    elif x.ndim == 4:
        if x.shape[1] == 3:
            pass
        elif x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Expected NCHW or NHWC batch, got shape {tuple(x.shape)}")
    else:
        raise ValueError(f"Expected image or image batch, got shape {tuple(x.shape)}")

    return x.contiguous()


class ImageNetModel:
    def __init__(self, model, device=None, model_source="auto"):
        self.device = _resolve_device(device)
        self.model_id = str(model)
        self.model, self.model_source, self.model_name = _load_pretrained_model(model, model_source)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.mu = torch.tensor([0.485, 0.456, 0.406], device=self.device).float().view(1, 3, 1, 1)
        self.sigma = torch.tensor([0.229, 0.224, 0.225], device=self.device).float().view(1, 3, 1, 1)

    def predict(self, x):
        x = _as_batched_chw(x, self.device)
        with torch.inference_mode():
            if self.model_source == "bcos":
                out = torch.cat([x, 1.0 - x], dim=1)
            else:
                out = (x - self.mu) / self.sigma
            return self.model(out)

    def forward(self, x):
        return self.predict(x)

    def __call__(self, x):
        return self.predict(x)


class RNDImageNet:
    def __init__(self, idx, device=None, model_source="auto"):
        self.model = ImageNetModel(idx, device=device, model_source=model_source)
        self.v = 0.02

    def predict(self, x):
        #x_ = x + np.random.normal(0, 1, size=x.shape) * self.v
        x = _as_batched_chw(x, self.model.device)
        x_ = x + torch.normal(mean=torch.zeros_like(x), std=torch.ones_like(x)) * self.v
        return self.model.predict(x_)
