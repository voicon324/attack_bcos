#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
    cat <<'EOF'
Usage:
  scripts/run_whitebox_spearman_sweep.sh [extra args...]

Runs `whitebox_spearman_experiment.py` on 10 B-cos models and 6
Linf PGD epsilon values:
  epsilon = 1/256
  epsilon = 2/256
  epsilon = 4/256
  epsilon = 8/256
  epsilon = 16/256
  epsilon = 32/256

Fixed by this wrapper:
  --images-csv, --model, --output-csv, --artifacts-dir,
  --attack-epsilon, --attack-step-size

Useful environment overrides:
  PYTHON_BIN=python
  DEVICE=cuda
  CSV_PATH=/path/to/used_images_500.csv
  BCOS_WEIGHTS_DIR=/kaggle/input/.../weights/bcos-imagenet
  LIMIT=500
  ATTACK_STEPS=20
  RESULT_ROOT=/path/to/output/root
  COMPONENT=alpha
  ALPHA_PERCENTILES="99.5"
  CLASS_KEEP_WEIGHT=10
  CLASS_KEEP_MARGIN=0
  FAST_RUNTIME=1
  SAVE_ARTIFACTS=1   # default; set SAVE_ARTIFACTS=0 to disable image outputs

Examples:
  scripts/run_whitebox_spearman_sweep.sh
  LIMIT=50 ATTACK_STEPS=10 scripts/run_whitebox_spearman_sweep.sh --verbose
  ALPHA_PERCENTILES="95 99 99.5" FAST_RUNTIME=1 scripts/run_whitebox_spearman_sweep.sh
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

TARGET_PATH="$ROOT_DIR/attacks/whitebox_spearman_experiment.py"
if [[ ! -f "$TARGET_PATH" ]]; then
    echo "Target script not found: $TARGET_PATH" >&2
    exit 1
fi

CSV_PATH="${CSV_PATH:-$ROOT_DIR/data/used_images_500.csv}"
if [[ ! -f "$CSV_PATH" ]]; then
    echo "CSV not found: $CSV_PATH" >&2
    exit 1
fi

DEVICE="${DEVICE:-cuda}"
LIMIT="${LIMIT:-500}"
ATTACK_STEPS="${ATTACK_STEPS:-20}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/artifacts/outputs/result/whitebox_spearman_model_epsilon_sweep}"
COMPONENT="${COMPONENT:-alpha}"
ALPHA_PERCENTILES="${ALPHA_PERCENTILES:-99.5}"
CLASS_KEEP_WEIGHT="${CLASS_KEEP_WEIGHT:-10}"
CLASS_KEEP_MARGIN="${CLASS_KEEP_MARGIN:-0}"
FAST_RUNTIME="${FAST_RUNTIME:-0}"
SAVE_ARTIFACTS="${SAVE_ARTIFACTS:-1}"

if [[ -z "${BCOS_WEIGHTS_DIR:-}" ]]; then
    for candidate in \
        "$ROOT_DIR/weights" \
        "$ROOT_DIR/weights/bcos-imagenet" \
        "$ROOT_DIR/artifacts/model-weights/weights" \
        "$ROOT_DIR/artifacts/model-weights/weights/bcos-imagenet" \
        "/kaggle/working/weights" \
        "/kaggle/working/weights/bcos-imagenet" \
        "/kaggle/working/bcos-weights" \
        "/kaggle/working/bcos-imagenet" \
        /kaggle/input/* \
        /kaggle/input/*/bcos-imagenet \
        /kaggle/input/*/weights \
        /kaggle/input/*/weights/bcos-imagenet \
        /kaggle/input/*/bcos-weights; do
        [[ -d "$candidate" ]] || continue
        if [[ -f "$candidate/resnet_18-68b4160fff.pth" ]]; then
            BCOS_WEIGHTS_DIR="$candidate"
            export BCOS_WEIGHTS_DIR
            break
        fi
    done
fi

read -r -a ALPHA_ARGS <<< "$ALPHA_PERCENTILES"

MODELS=(
    resnet18
    resnet34
    resnet50
    resnet101
    resnet152
    resnext50_32x4d
    densenet121
    densenet161
    densenet169
    densenet201
)

EPS_LABELS=("1_256" "2_256" "4_256" "8_256" "16_256" "32_256")
EPS_DISPLAY=("1/256" "2/256" "4/256" "8/256" "16/256" "32/256")
EPS_VALUES=("0.00390625" "0.0078125" "0.015625" "0.03125" "0.0625" "0.125")
STEP_VALUES=("0.0009765625" "0.001953125" "0.00390625" "0.0078125" "0.015625" "0.03125")

COMMON_ARGS=(
    --images-csv "$CSV_PATH"
    --device "$DEVICE"
    --limit "$LIMIT"
    --attack-steps "$ATTACK_STEPS"
    --component "$COMPONENT"
    --alpha-percentiles "${ALPHA_ARGS[@]}"
    --class-keep-weight "$CLASS_KEEP_WEIGHT"
    --class-keep-margin "$CLASS_KEEP_MARGIN"
)

if [[ "$FAST_RUNTIME" == "1" || "$FAST_RUNTIME" == "true" || "$FAST_RUNTIME" == "yes" ]]; then
    COMMON_ARGS+=(--fast-runtime)
fi

if [[ "$SAVE_ARTIFACTS" == "1" || "$SAVE_ARTIFACTS" == "true" || "$SAVE_ARTIFACTS" == "yes" ]]; then
    COMMON_ARGS+=(--save-artifacts)
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --images-csv|--images-csv=*|--model|--model=*|--output-csv|--output-csv=*|--artifacts-dir|--artifacts-dir=*|--attack-epsilon|--attack-epsilon=*|--attack-step-size|--attack-step-size=*)
            echo "Do not pass $1 to this sweep wrapper; that option is managed per run." >&2
            exit 1
            ;;
        *)
            COMMON_ARGS+=("$1")
            shift
            ;;
    esac
done

mkdir -p "$RESULT_ROOT"

total_runs=$((${#MODELS[@]} * ${#EPS_VALUES[@]}))
run_index=0

echo "================================================================"
echo "Whitebox explain Spearman sweep"
echo "Models       : ${MODELS[*]}"
echo "Epsilons     : ${EPS_DISPLAY[*]}"
echo "CSV          : $CSV_PATH"
echo "Limit        : $LIMIT"
echo "Attack steps : $ATTACK_STEPS"
echo "Device       : $DEVICE"
echo "Keep cls wgt : $CLASS_KEEP_WEIGHT"
echo "Keep margin  : $CLASS_KEEP_MARGIN"
echo "Save images  : $SAVE_ARTIFACTS"
echo "Weights dir  : ${BCOS_WEIGHTS_DIR:-<auto-download/fallback>}"
echo "Result root  : $RESULT_ROOT"
echo "================================================================"

for model in "${MODELS[@]}"; do
    for i in "${!EPS_VALUES[@]}"; do
        run_index=$((run_index + 1))
        eps_label="${EPS_LABELS[$i]}"
        eps_display="${EPS_DISPLAY[$i]}"
        eps_value="${EPS_VALUES[$i]}"
        step_value="${STEP_VALUES[$i]}"
        run_dir="$RESULT_ROOT/${model}/eps_${eps_label}"
        output_csv="$run_dir/results.csv"
        artifacts_dir="$run_dir/artifacts"

        mkdir -p "$run_dir"

        echo
        echo "================================================================"
        echo "Run $run_index/$total_runs"
        echo "Model        : $model"
        echo "Epsilon      : $eps_display ($eps_value)"
        echo "Step size    : $step_value"
        echo "Output CSV   : $output_csv"
        echo "================================================================"

        "$PYTHON_BIN" "$TARGET_PATH" \
            "${COMMON_ARGS[@]}" \
            --model "$model" \
            --attack-epsilon "$eps_value" \
            --attack-step-size "$step_value" \
            --output-csv "$output_csv" \
            --artifacts-dir "$artifacts_dir"
    done
done
