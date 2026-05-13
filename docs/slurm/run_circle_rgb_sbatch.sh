#!/usr/bin/env bash
#SBATCH --job-name=bcos-camo
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=mps:a100:2
#SBATCH --mem=16G
#SBATCH --time=72:00:00

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-bcos}"
CONDA_SH="${CONDA_SH:-}"
VENV_PATH="${VENV_PATH-$ROOT/.venv}"
REQUIRED_VRAM="${REQUIRED_VRAM:-16000}"

echo "Job ID       : ${SLURM_JOB_ID:-local}"
echo "Node         : ${SLURMD_NODENAME:-unknown}"
echo "Repo         : $ROOT"
echo "Conda env    : $CONDA_ENV"
echo "Venv path    : ${VENV_PATH:-disabled}"
echo "Required VRAM: $REQUIRED_VRAM MiB"

if command -v module >/dev/null 2>&1; then
  module clear -f || true
fi

USE_VENV=0
if [[ -n "$VENV_PATH" && -f "$VENV_PATH/bin/activate" ]]; then
  source "$VENV_PATH/bin/activate"
  USE_VENV=1
  echo "Using venv   : $VENV_PATH"
else
  if [[ -n "$CONDA_SH" && -f "$CONDA_SH" ]]; then
    source "$CONDA_SH"
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    echo "Could not find a .venv or conda.sh. Set VENV_PATH=/path/to/.venv or CONDA_SH=/path/to/conda.sh." >&2
    exit 1
  fi
  echo "Using conda  : $CONDA_ENV"
fi

if [[ -x /usr/local/bin/gpu_check.sh && -n "${SLURM_JOB_ID:-}" ]]; then
  unset CUDA_VISIBLE_DEVICES
  set +e
  CHECK_OUT=$(/usr/local/bin/gpu_check.sh "$REQUIRED_VRAM" "$SLURM_JOB_ID")
  EXIT_CODE=$?
  set -e
  if [[ "$EXIT_CODE" -eq 10 ]]; then
    echo "$CHECK_OUT"
    exit 0
  elif [[ "$EXIT_CODE" -eq 11 ]]; then
    echo "$CHECK_OUT"
    exit 1
  fi

  BEST_GPU="$CHECK_OUT"
  export CUDA_VISIBLE_DEVICES="$BEST_GPU"
  export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-job${SLURM_JOB_ID}"
  export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-job${SLURM_JOB_ID}"
  rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  echo "Selected GPU : $BEST_GPU"
else
  echo "CUDA devices : ${CUDA_VISIBLE_DEVICES:-not set by Slurm}"
fi

if [[ "$USE_VENV" == "1" ]]; then
  python - <<'PY'
import torch
print("Python executable    :", __import__("sys").executable)
print("Python CUDA available:", torch.cuda.is_available())
print("Python CUDA devices  :", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Python GPU name      :", torch.cuda.get_device_name(0))
PY
else
  conda run --no-capture-output -n "$CONDA_ENV" python - <<'PY'
import torch
print("Python executable    :", __import__("sys").executable)
print("Python CUDA available:", torch.cuda.is_available())
print("Python CUDA devices  :", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Python GPU name      :", torch.cuda.get_device_name(0))
PY
fi

bash "$ROOT/scripts/run_circle_rgb_local.sh"
