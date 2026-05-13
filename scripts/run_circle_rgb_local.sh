#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CSV_PATH="${CSV_PATH:-$ROOT/data/used_images_500_local.csv}"
CONDA_ENV="${CONDA_ENV:-bcos}"
VENV_PATH="${VENV_PATH-$ROOT/.venv}"
MODEL="${MODEL:-resnet50}"
DEVICE="${DEVICE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/artifacts/outputs/result/explain_guided_circle_rgb_outputs_csv}"

PATCH_SIZE="${PATCH_SIZE:-16}"
NUM_CIRCLES="${NUM_CIRCLES:-8}"
GENERATIONS="${GENERATIONS:-2000}"
OPTIMIZER="${OPTIMIZER:-camo-1p1}"
CAMO_MUT="${CAMO_MUT:-0.3}"
POPULATION="${POPULATION:-5}"
IMAGE_BATCH_SIZE="${IMAGE_BATCH_SIZE:-8}"
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

if [[ -d "$ROOT/artifacts/model-weights/weights" && -z "${MODEL_WEIGHTS_DIR:-}" ]]; then
  export MODEL_WEIGHTS_DIR="$ROOT/artifacts/model-weights/weights"
fi

if [[ -d "$ROOT/artifacts/model-weights/weights/bcos-imagenet" && -z "${BCOS_WEIGHTS_DIR:-}" ]]; then
  export BCOS_WEIGHTS_DIR="$ROOT/artifacts/model-weights/weights/bcos-imagenet"
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

if [[ -n "$VENV_PATH" && -f "$VENV_PATH/bin/activate" ]]; then
  source "$VENV_PATH/bin/activate"
  echo "Using venv: $VENV_PATH"
  echo "Python    : $(command -v python)"
  "${args[@]}"
else
  echo "Using conda env: $CONDA_ENV"
  conda run --no-capture-output -n "$CONDA_ENV" "${args[@]}"
fi
