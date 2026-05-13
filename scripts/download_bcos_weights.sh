#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/artifacts/model-weights/weights/bcos-imagenet}"

mkdir -p "$OUT_DIR"

python3 - "$OUT_DIR" <<'PY'
import sys
from pathlib import Path
from urllib.request import urlretrieve

urls = [
    "https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/resnet_18-68b4160fff.pth",
    "https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/resnet_34-a63425a03e.pth",
    "https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/resnet_50-ead259efe4.pth",
    "https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/resnet_101-84c3658278.pth",
    "https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/resnet_152-42051a77c1.pth",
    "https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/resnext_50_32x4d-57af241ab9.pth",
    "https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/densenet_121-b8daf96afb.pth",
    "https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/densenet_161-9e9ea51353.pth",
    "https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/densenet_169-7037ee0604.pth",
    "https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/densenet_201-00ac87066f.pth",
]

out_dir = Path(sys.argv[1])
out_dir.mkdir(parents=True, exist_ok=True)

for url in urls:
    filename = Path(url).name
    destination = out_dir / filename
    if destination.exists():
        print(f"skip  {destination}")
        continue
    print(f"fetch {url}")
    urlretrieve(url, destination)
    print(f"saved {destination}")
PY
