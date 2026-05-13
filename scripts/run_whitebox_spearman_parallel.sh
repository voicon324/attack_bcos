#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_SWEEP="$ROOT_DIR/scripts/run_whitebox_spearman_sweep.sh"

usage() {
    cat <<'EOF'
Usage:
  scripts/run_whitebox_spearman_parallel.sh [extra args...]

Runs run_whitebox_spearman_sweep.sh on multiple image chunks at
the same time. Each chunk gets a different --start/--limit range and writes to
its own result directory, then CSV files are merged by model/epsilon.

Useful environment overrides:
  IMAGE_JOBS=2       Number of image chunks to run at the same time.
  TOTAL_LIMIT=500    Total number of images across all chunks.
  START=0            First valid CSV row offset.
  RESULT_ROOT=...    Top-level output directory for chunks and merged CSVs.
  DEVICE=cuda        Device used by every chunk when CUDA_DEVICES is not set.
  CUDA_DEVICES="cuda:0 cuda:1"
                     Optional round-robin device list, one per chunk.
  MERGE_RESULTS=1    Merge chunk CSVs into RESULT_ROOT/merged after success.

All other environment variables supported by the base sweep script still work:
  PYTHON_BIN, CSV_PATH, ATTACK_STEPS, COMPONENT, ALPHA_PERCENTILES,
  CLASS_KEEP_WEIGHT, CLASS_KEEP_MARGIN, FAST_RUNTIME, SAVE_ARTIFACTS

Do not pass --start or --limit as extra args; use START and TOTAL_LIMIT here.

Examples:
  IMAGE_JOBS=4 TOTAL_LIMIT=500 scripts/run_whitebox_spearman_parallel.sh
  CUDA_DEVICES="cuda:0 cuda:1" IMAGE_JOBS=2 TOTAL_LIMIT=500 scripts/run_whitebox_spearman_parallel.sh --verbose
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

if [[ ! -x "$BASE_SWEEP" ]]; then
    echo "Base sweep script not found or not executable: $BASE_SWEEP" >&2
    exit 1
fi

for arg in "$@"; do
    case "$arg" in
        --start|--start=*|--limit|--limit=*)
            echo "Do not pass $arg to this parallel wrapper; use START and TOTAL_LIMIT." >&2
            exit 1
            ;;
    esac
done

IMAGE_JOBS="${IMAGE_JOBS:-2}"
TOTAL_LIMIT="${TOTAL_LIMIT:-${LIMIT:-500}}"
START="${START:-0}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/artifacts/outputs/result/whitebox_spearman_model_epsilon_sweep_parallel}"
MERGE_RESULTS="${MERGE_RESULTS:-1}"

if ! [[ "$IMAGE_JOBS" =~ ^[0-9]+$ ]] || (( IMAGE_JOBS <= 0 )); then
    echo "IMAGE_JOBS must be a positive integer, got: $IMAGE_JOBS" >&2
    exit 1
fi
if ! [[ "$TOTAL_LIMIT" =~ ^[0-9]+$ ]] || (( TOTAL_LIMIT <= 0 )); then
    echo "TOTAL_LIMIT must be a positive integer, got: $TOTAL_LIMIT" >&2
    exit 1
fi
if ! [[ "$START" =~ ^[0-9]+$ ]]; then
    echo "START must be a non-negative integer, got: $START" >&2
    exit 1
fi

if [[ -n "${CUDA_DEVICES:-}" ]]; then
    read -r -a DEVICE_LIST <<< "$CUDA_DEVICES"
elif [[ -n "${DEVICE:-}" ]]; then
    DEVICE_LIST=("$DEVICE")
else
    DEVICE_LIST=("cuda")
fi

if (( ${#DEVICE_LIST[@]} == 0 )); then
    echo "CUDA_DEVICES did not contain any devices." >&2
    exit 1
fi

chunk_size=$(((TOTAL_LIMIT + IMAGE_JOBS - 1) / IMAGE_JOBS))
chunks_dir="$RESULT_ROOT/chunks"
logs_dir="$RESULT_ROOT/logs"
merged_dir="$RESULT_ROOT/merged"

mkdir -p "$chunks_dir" "$logs_dir"

echo "================================================================"
echo "Whitebox explain Spearman parallel image sweep"
echo "Image jobs   : $IMAGE_JOBS"
echo "Total limit  : $TOTAL_LIMIT"
echo "Start        : $START"
echo "Chunk size   : $chunk_size"
echo "Devices      : ${DEVICE_LIST[*]}"
echo "Result root  : $RESULT_ROOT"
echo "Logs         : $logs_dir"
echo "================================================================"

pids=()
chunk_ids=()

for ((job = 0; job < IMAGE_JOBS; job++)); do
    chunk_start=$((START + job * chunk_size))
    remaining=$((START + TOTAL_LIMIT - chunk_start))
    if (( remaining <= 0 )); then
        break
    fi

    chunk_limit=$chunk_size
    if (( remaining < chunk_limit )); then
        chunk_limit=$remaining
    fi

    chunk_id="$(printf "chunk_%03d" "$job")"
    chunk_root="$chunks_dir/$chunk_id"
    log_path="$logs_dir/$chunk_id.log"
    device="${DEVICE_LIST[$((job % ${#DEVICE_LIST[@]}))]}"

    echo "Starting $chunk_id: start=$chunk_start limit=$chunk_limit device=$device"
    (
        set -euo pipefail
        DEVICE="$device" \
        LIMIT="$chunk_limit" \
        RESULT_ROOT="$chunk_root" \
        "$BASE_SWEEP" --start "$chunk_start" "$@"
    ) > "$log_path" 2>&1 &

    pids+=("$!")
    chunk_ids+=("$chunk_id")
done

if (( ${#pids[@]} == 0 )); then
    echo "No chunks were started." >&2
    exit 1
fi

status=0
for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    chunk_id="${chunk_ids[$i]}"
    if wait "$pid"; then
        echo "Finished $chunk_id"
    else
        echo "Failed $chunk_id; see $logs_dir/$chunk_id.log" >&2
        status=1
    fi
done

if (( status != 0 )); then
    exit "$status"
fi

merge_results() {
    shopt -s nullglob
    mkdir -p "$merged_dir"

    mapfile -t rel_paths < <(
        find "$chunks_dir" -mindepth 4 -maxdepth 4 -type f -name results.csv -printf '%P\n' \
            | sed -E 's#^chunk_[0-9]+/##' \
            | sort -u
    )

    if (( ${#rel_paths[@]} == 0 )); then
        echo "No result CSV files found to merge." >&2
        return 1
    fi

    for rel_path in "${rel_paths[@]}"; do
        out_path="$merged_dir/$rel_path"
        mkdir -p "$(dirname "$out_path")"
        tmp_path="$out_path.tmp"
        first=1
        : > "$tmp_path"

        for src_path in "$chunks_dir"/chunk_*/"$rel_path"; do
            [[ -f "$src_path" ]] || continue
            if (( first )); then
                head -n 1 "$src_path" >> "$tmp_path"
                first=0
            fi
            tail -n +2 "$src_path" >> "$tmp_path"
        done

        mv "$tmp_path" "$out_path"
        echo "Merged $rel_path"
    done
}

if [[ "$MERGE_RESULTS" == "1" || "$MERGE_RESULTS" == "true" || "$MERGE_RESULTS" == "yes" ]]; then
    echo
    echo "Merging result CSV files into $merged_dir"
    merge_results
fi

echo
echo "Done."
echo "Chunk outputs : $chunks_dir"
echo "Merged CSVs   : $merged_dir"
echo "Logs          : $logs_dir"
