#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/artifacts/model-weights/weights/torchvision-imagenet}"

mkdir -p "$OUT_DIR"

python3 - "$OUT_DIR" <<'PY'
import sys
from pathlib import Path
from urllib.request import urlretrieve

urls = [
    "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
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
