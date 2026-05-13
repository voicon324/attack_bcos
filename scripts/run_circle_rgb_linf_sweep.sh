#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

POPULATION="${POPULATION:-5}"
GENERATIONS="${GENERATIONS:-2000}"
OPTIMIZER="${OPTIMIZER:-camo-1p1}"
CAMO_MUT="${CAMO_MUT:-0.3}"
ATTACK_ORIGINAL_MODEL="${ATTACK_ORIGINAL_MODEL:-1}"

if [[ -z "${RUNNER:-}" ]]; then
  if [[ "$ROOT" == /kaggle/input/* || -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]]; then
    RUNNER="$ROOT/scripts/run_circle_rgb_kaggle.sh"
  else
    RUNNER="$ROOT/scripts/run_circle_rgb_local.sh"
  fi
fi

if [[ ! -f "$RUNNER" ]]; then
  echo "Runner not found: $RUNNER" >&2
  exit 1
fi

if [[ -z "${OUTPUT_ROOT:-}" ]]; then
  if [[ "$ROOT" == /kaggle/input/* || -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]]; then
    OUTPUT_ROOT="/kaggle/working/result/explain_guided_circle_rgb_linf_sweep"
  else
    OUTPUT_ROOT="$ROOT/artifacts/outputs/result_circle/explain_guided_circle_rgb_linf_sweep"
  fi
fi

EPS_LABELS=("16_256" "32_256" "64_256")
EPS_DISPLAY=("16/256" "32/256" "64/256")
EPS_VALUES=("0.0625" "0.125" "0.25")

echo "========================================================================"
echo "  Circle RGB Linf Sweep"
echo "========================================================================"
echo "  Runner      : $RUNNER"
echo "  Attack mode : $([[ "$ATTACK_ORIGINAL_MODEL" == "1" ]] && echo torchvision-original || echo bcos)"
echo "  Optimizer   : $OPTIMIZER"
echo "  Queries     : $GENERATIONS"
echo "  Population  : $POPULATION"
echo "  Output root : $OUTPUT_ROOT"
echo

for i in "${!EPS_VALUES[@]}"; do
  eps_label="${EPS_LABELS[$i]}"
  eps_display="${EPS_DISPLAY[$i]}"
  eps_value="${EPS_VALUES[$i]}"
  run_dir="$OUTPUT_ROOT/linf_${eps_label}"
  rgb_epsilon="${RGB_EPSILON:-$eps_value}"

  mkdir -p "$run_dir"

  echo "========================================================================"
  echo "  Running Linf = $eps_display ($eps_value)"
  echo "  RGB epsilon  = $rgb_epsilon"
  echo "  Output dir   = $run_dir"
  echo "========================================================================"

  POPULATION="$POPULATION" \
  GENERATIONS="$GENERATIONS" \
  OPTIMIZER="$OPTIMIZER" \
  CAMO_MUT="$CAMO_MUT" \
  ATTACK_ORIGINAL_MODEL="$ATTACK_ORIGINAL_MODEL" \
  LINF_EPSILON="$eps_value" \
  RGB_EPSILON="$rgb_epsilon" \
  OUTPUT_DIR="$run_dir" \
  bash "$RUNNER"
done
