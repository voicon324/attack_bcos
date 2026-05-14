#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python}"
model_source="${MODEL_SOURCE:-torchvision}"
linfs=("16/256" "32/256" "64/256")

maybe_set_weights_dir() {
  if [ -n "${MODEL_WEIGHTS_DIR:-}" ] || [ -n "${TORCHVISION_WEIGHTS_DIR:-}" ]; then
    return
  fi

  local candidate=""
  for candidate in \
    "/kaggle/input/weights-bcos" \
    "/kaggle/input/datasets/hkhnhduy/weights-bcos"; do
    if [ -e "$candidate/torchvision-imagenet.zip" ] || [ -d "$candidate/torchvision-imagenet" ]; then
      export MODEL_WEIGHTS_DIR="$candidate"
      echo "Using MODEL_WEIGHTS_DIR=$MODEL_WEIGHTS_DIR"
      return
    fi
  done

  if [ -d "/kaggle/input" ]; then
    while IFS= read -r candidate; do
      export MODEL_WEIGHTS_DIR="$candidate"
      echo "Using MODEL_WEIGHTS_DIR=$MODEL_WEIGHTS_DIR"
      return
    done < <(find /kaggle/input -maxdepth 2 \( -name torchvision-imagenet.zip -o -name torchvision-imagenet \) -print 2>/dev/null | xargs -r -n1 dirname | sort -u)
  fi
}

maybe_set_weights_dir

usage() {
  cat <<EOF
Usage:
  $0 CSV_PATH SAVE_ROOT [extra ConCamoPatch args...]
  $0 IMAGE_DIR TRUE_LABEL SAVE_PREFIX [extra ConCamoPatch args...]

Examples:
  $0 ../used_images_500_1.csv results/camopatch_resnet50 --device cuda --batch_size 8 --trace_every 0
  MODEL_SOURCE=bcos $0 ../used_images_500_1.csv results/camopatch_bcos_resnet50 --device cuda
  $0 ./8.JPEG 8 results/8 --device cuda --batch_size 8 --trace_every 0

MODEL_SOURCE defaults to torchvision for original ResNet-50. Set MODEL_SOURCE=bcos for B-cos ResNet-50.
EOF
}

run_one() {
  local image_dir="$1"
  local true_label="$2"
  local save_prefix="$3"
  shift 3

  for linf in "${linfs[@]}"; do
    local linf_suffix="${linf//\//_}"
    local save_directory="${save_prefix}_resnet50_${model_source}_linf_${linf_suffix}"

    echo "Running ResNet-50 (${model_source}) with --linf ${linf}"
    echo "Image: ${image_dir}"
    echo "Label: ${true_label}"
    echo "Saving to ${save_directory}.npy"

    "$python_bin" "$script_dir/ConCamoPatch.py" \
      --model 1 \
      --model_source "$model_source" \
      --image_dir "$image_dir" \
      --true_label "$true_label" \
      --save_directory "$save_directory" \
      --linf "$linf" \
      "$@"
  done
}

if [ "$#" -lt 2 ]; then
  usage
  exit 1
fi

first_arg="$1"

if [[ "$first_arg" == *.csv ]]; then
  csv_path="$1"
  save_root="$2"
  shift 2

  mkdir -p "$save_root"

  while IFS=$'\t' read -r row_idx true_label image_stem image_path; do
    image_save_prefix="${save_root}/${row_idx}_${image_stem}_label_${true_label}"
    run_one "$image_path" "$true_label" "$image_save_prefix" "$@"
  done < <("$python_bin" - "$csv_path" <<'PY'
import csv
import re
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
label_columns = ("true_label", "label", "class_idx", "class_id", "target_class", "pred_class")

with csv_path.open(newline="") as f:
    reader = csv.DictReader(f)
    if not reader.fieldnames:
        raise SystemExit(f"CSV has no header: {csv_path}")
    if "image_path" in reader.fieldnames:
        image_column = "image_path"
    else:
        image_column = reader.fieldnames[0]

    label_column = next((name for name in label_columns if name in reader.fieldnames), None)

    for idx, row in enumerate(reader, start=1):
        image_path = (row.get(image_column) or "").strip()
        if not image_path:
            continue

        if label_column and (row.get(label_column) or "").strip() != "":
            true_label = int(float(row[label_column]))
        else:
            parent = Path(image_path).parent.name
            if not parent.isdigit():
                raise SystemExit(
                    f"Cannot infer label for row {idx}: no label column and parent directory is not numeric ({parent!r})"
                )
            true_label = int(parent)

        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(image_path).stem).strip("._")
        row_name = f"{idx:05d}"
        print(f"{row_name}\t{true_label}\t{stem}\t{image_path}")
PY
  )
else
  if [ "$#" -lt 3 ]; then
    usage
    exit 1
  fi

  image_dir="$1"
  true_label="$2"
  save_prefix="$3"
  shift 3
  run_one "$image_dir" "$true_label" "$save_prefix" "$@"
fi
