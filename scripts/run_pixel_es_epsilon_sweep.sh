#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
    cat <<'EOF'
Usage:
  scripts/run_pixel_es_epsilon_sweep.sh [extra args...]

Runs `explain_guided_pixel_es_patch.py` on all images in `data/used_images_500.csv`
with the B-cos model `resnet50`, sweeping:
  epsilon = 8/128
  epsilon = 16/128
  epsilon = 32/128

Notes:
  - Do not pass --epsilon yourself; this wrapper injects it.
  - Do not pass --image, --images-csv, --model, or --output-dir; they are fixed here.
  - Each run writes into its own directory under ./artifacts/outputs/result/runs_linf/
    so outputs do not overwrite each other.

Examples:
  scripts/run_pixel_es_epsilon_sweep.sh --norm linf
  scripts/run_pixel_es_epsilon_sweep.sh --norm l2 --patch-size 32
  scripts/run_pixel_es_epsilon_sweep.sh --norm linf --save-images --save-figure
EOF
}

if [[ $# -gt 0 ]]; then
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
    esac
fi

TARGET_PATH="$ROOT_DIR/attacks/explain_guided_pixel_es_patch.py"
RESULT_ROOT="$ROOT_DIR/artifacts/outputs/result"

if [[ ! -f "$TARGET_PATH" ]]; then
    echo "Target script not found: $TARGET_PATH" >&2
    exit 1
fi

CSV_PATH="$ROOT_DIR/data/used_images_500.csv"
if [[ ! -f "$CSV_PATH" ]]; then
    echo "CSV not found: $CSV_PATH" >&2
    exit 1
fi

NORMALIZED_ARGS=(
    --images-csv "$CSV_PATH"
    --model resnet50
    --device cuda
    --amp-dtype bfloat16
)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epsilon|--epsilon=*|--image|--image=*|--images-csv|--images-csv=*|--model|--model=*|--output-dir|--output-dir=*)
            echo "Do not pass $1 to run_pixel_es_epsilon_sweep.sh; epsilon/images/model are fixed in this wrapper." >&2
            exit 1
            ;;
        *)
            NORMALIZED_ARGS+=("$1")
            shift
            ;;
    esac
done

EPS_LABELS=("8_128" "16_128" "32_128")
EPS_DISPLAY=("8/128" "16/128" "32/128")
EPS_VALUES=("0.0625" "0.125" "0.25")

TARGET_NAME="$(basename "$TARGET_PATH" .py)"

for i in "${!EPS_VALUES[@]}"; do
    eps_label="${EPS_LABELS[$i]}"
    eps_display="${EPS_DISPLAY[$i]}"
    eps_value="${EPS_VALUES[$i]}"
    run_dir="$RESULT_ROOT/runs_linf/${TARGET_NAME}_eps_${eps_label}"

    mkdir -p "$run_dir"

    echo "================================================================"
    echo "Running $TARGET_NAME with epsilon = $eps_display ($eps_value)"
    echo "Work dir: $run_dir"
    echo "================================================================"

    "$PYTHON_BIN" "$TARGET_PATH" "${NORMALIZED_ARGS[@]}" --epsilon "$eps_value" --output-dir "$run_dir"
done
