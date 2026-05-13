#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d "$ROOT/B-cos-v2" && -f "$ROOT/B-cos-v2.zip" && ! -d "/kaggle/working/B-cos-v2" ]]; then
  echo "Extracting B-cos-v2.zip to /kaggle/working ..."
  unzip -q "$ROOT/B-cos-v2.zip" -d /kaggle/working
fi

MODEL="${MODEL:-resnet50}"
DEVICE="${DEVICE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-/kaggle/working/result/explain_guided_circle_rgb_outputs_csv}"

PATCH_SIZE="${PATCH_SIZE:-16}"
NUM_CIRCLES="${NUM_CIRCLES:-8}"
GENERATIONS="${GENERATIONS:-2000}"
OPTIMIZER="${OPTIMIZER:-camo-1p1}"
CAMO_MUT="${CAMO_MUT:-0.3}"
POPULATION="${POPULATION:-5}"
IMAGE_BATCH_SIZE="${IMAGE_BATCH_SIZE:-1}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-0}"

MIN_RADIUS="${MIN_RADIUS:-1}"
MAX_RADIUS="${MAX_RADIUS:-8}"
MIN_ALPHA="${MIN_ALPHA:-0}"
MAX_ALPHA="${MAX_ALPHA:-1}"
RGB_EPSILON="${RGB_EPSILON:-0.3}"
LINF_EPSILON="${LINF_EPSILON:-$RGB_EPSILON}"
EDGE_SOFTNESS="${EDGE_SOFTNESS:-1}"

SAVE_IMAGES="${SAVE_IMAGES:-0}"
SAVE_FIGURE="${SAVE_FIGURE:-0}"
VERBOSE_GENERATIONS="${VERBOSE_GENERATIONS:-0}"
ATTACK_ORIGINAL_MODEL="${ATTACK_ORIGINAL_MODEL:-1}"

find_bcos_weights_dir() {
  if [[ -n "${BCOS_WEIGHTS_DIR:-}" && -d "$BCOS_WEIGHTS_DIR" ]]; then
    printf '%s\n' "$BCOS_WEIGHTS_DIR"
    return 0
  fi

  local candidates=(
    "/kaggle/input/weights-bcos/bcos-imagenet"
    "/kaggle/input/weights-bcos"
    "/kaggle/input/datasets/hkhnhduy/weights-bcos/bcos-imagenet"
    "/kaggle/input/datasets/hkhnhduy/weights-bcos"
    "/kaggle/input/datasets/weights-bcos/bcos-imagenet"
    "/kaggle/input/datasets/weights-bcos"
    "/kaggle/working/weights/bcos-imagenet"
    "/kaggle/working/weights"
    "$ROOT/weights/bcos-imagenet"
    "$ROOT/weights"
    "$ROOT/artifacts/model-weights/weights/bcos-imagenet"
    "$ROOT/artifacts/model-weights/weights"
  )
  local candidate=""
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate" ]] && find "$candidate" -maxdepth 2 -type f -name '*.pth' -print -quit | grep -q .; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  while IFS= read -r candidate; do
    printf '%s\n' "$(dirname "$candidate")"
    return 0
  done < <(find /kaggle/input /kaggle/working -maxdepth 7 -type f -name 'resnet_50-ead259efe4.pth' 2>/dev/null | sort)

  while IFS= read -r candidate; do
    printf '%s\n' "$(dirname "$candidate")"
    return 0
  done < <(find /kaggle/input /kaggle/working -maxdepth 7 -type f -path '*/bcos-imagenet/*.pth' 2>/dev/null | sort)

  return 1
}

find_imagenet_root() {
  if [[ -n "${IMAGENET_ROOT:-}" && -d "$IMAGENET_ROOT" ]]; then
    printf '%s\n' "$IMAGENET_ROOT"
    return 0
  fi

  local candidates=(
    "/kaggle/input/imagenet1kvalid"
    "/kaggle/input/datasets/sautkin/imagenet1kvalid"
  )
  local candidate=""
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  while IFS= read -r candidate; do
    if [[ -d "$candidate/00000" || -d "$candidate/00482" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done < <(find /kaggle/input -maxdepth 4 -type d -name 'imagenet1kvalid' 2>/dev/null | sort)

  return 1
}

CSV_PATH="${CSV_PATH:-}"
if [[ -z "$CSV_PATH" ]]; then
  imagenet_root="$(find_imagenet_root)" || {
    echo "Could not find ImageNet root. Set IMAGENET_ROOT=/kaggle/input/<dataset>/imagenet1kvalid" >&2
    exit 1
  }
  CSV_PATH="/kaggle/working/used_images_500_kaggle_runtime.csv"
  REL_CSV="$ROOT/data/used_images_500_rel.csv"
  if [[ ! -f "$REL_CSV" ]]; then
    REL_CSV="$ROOT/used_images_500_rel.csv"
  fi
  awk -v root="$imagenet_root" 'NR==1 {print "image_path"; next} {gsub(/\r/, ""); print root "/" $1}' \
    "$REL_CSV" > "$CSV_PATH"
  echo "ImageNet root: $imagenet_root"
  echo "Runtime CSV  : $CSV_PATH"
fi

if bcos_weights_dir="$(find_bcos_weights_dir)"; then
  export BCOS_WEIGHTS_DIR="$bcos_weights_dir"
  export MODEL_WEIGHTS_DIR="${MODEL_WEIGHTS_DIR:-$bcos_weights_dir}"
  export WEIGHTS_DIR="${WEIGHTS_DIR:-$bcos_weights_dir}"
  echo "B-cos weights: $BCOS_WEIGHTS_DIR"
else
  echo "B-cos weights: not found locally; loader will try its own cache/search." >&2
fi

args=(
  python -u "$ROOT/attacks/explain_guided_circle_rgb_es_patch.py"
  --images-csv "$CSV_PATH"
  --model "$MODEL"
  --device "$DEVICE"
  --output-dir "$OUTPUT_DIR"
  --patch-size "$PATCH_SIZE"
  --num-circles "$NUM_CIRCLES"
  --generations "$GENERATIONS"
  --optimizer "$OPTIMIZER"
  --camo-mut "$CAMO_MUT"
  --population "$POPULATION"
  --image-batch-size "$IMAGE_BATCH_SIZE"
  --score-batch-size "$SCORE_BATCH_SIZE"
  --min-radius "$MIN_RADIUS"
  --max-radius "$MAX_RADIUS"
  --min-alpha "$MIN_ALPHA"
  --max-alpha "$MAX_ALPHA"
  --rgb-epsilon "$RGB_EPSILON"
  --linf-epsilon "$LINF_EPSILON"
  --edge-softness "$EDGE_SOFTNESS"
)

if [[ "$SAVE_IMAGES" == "1" ]]; then
  args+=(--save-images)
fi

if [[ "$SAVE_FIGURE" == "1" ]]; then
  args+=(--save-figure)
fi

if [[ "$VERBOSE_GENERATIONS" == "1" ]]; then
  args+=(--verbose-generations)
fi

if [[ "$ATTACK_ORIGINAL_MODEL" == "1" ]]; then
  args+=(--attack-original-model)
fi

"${args[@]}"
