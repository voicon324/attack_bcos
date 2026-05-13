#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" == "-n" || "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
else
    DRY_RUN=0
fi

KAGGLE_JSON="${KAGGLE_JSON:-$ROOT/artifacts/secrets/kaggle.json}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-${WEIGHTS_DIR:-$ROOT/artifacts/model-weights/weights}}"
DATASET_SLUG="${KAGGLE_WEIGHTS_DATASET_SLUG:-weights-bcos}"
DATASET_TITLE="${KAGGLE_WEIGHTS_DATASET_TITLE:-weights-bcos}"
KAGGLE_LICENSE="${KAGGLE_LICENSE:-CC0-1.0}"
KAGGLE_IS_PRIVATE="${KAGGLE_IS_PRIVATE:-true}"
KAGGLE_DIR_MODE="${KAGGLE_DIR_MODE:-zip}"
VERSION_NOTE="${1:-sync $(date '+%Y-%m-%d %H:%M:%S')}"

if [[ ! -f "$KAGGLE_JSON" ]]; then
    echo "Missing $KAGGLE_JSON" >&2
    exit 1
fi

if [[ ! -d "$WEIGHTS_ROOT" ]]; then
    echo "Weights directory not found: $WEIGHTS_ROOT" >&2
    exit 1
fi

if ! command -v git >/dev/null 2>&1; then
    echo "git is required" >&2
    exit 1
fi

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
else
    echo "python or python3 is required" >&2
    exit 1
fi

KAGGLE_CMD=()
CLI_VENV=""
if command -v kaggle >/dev/null 2>&1; then
    KAGGLE_CMD=("$(command -v kaggle)")
else
    echo "Installing kaggle into a temporary venv..."
    CLI_VENV="$(mktemp -d)"
    "$PYTHON_BIN" -m venv "$CLI_VENV"
    "$CLI_VENV/bin/pip" install --quiet --disable-pip-version-check kaggle
    KAGGLE_CMD=("$CLI_VENV/bin/kaggle")
fi

CFG_DIR="$(mktemp -d)"
STAGE_DIR="$(mktemp -d)"
META_DIR="$(mktemp -d)"
cleanup() {
    rm -rf "$CFG_DIR" "$STAGE_DIR" "$META_DIR" "$CLI_VENV"
}
trap cleanup EXIT

install -m 600 "$KAGGLE_JSON" "$CFG_DIR/kaggle.json"
export KAGGLE_CONFIG_DIR="$CFG_DIR"

USERNAME="$("$PYTHON_BIN" -c 'import json, sys; print(json.load(open(sys.argv[1]))["username"])' "$KAGGLE_JSON")"
DATASET_ID="${USERNAME}/${DATASET_SLUG}"
export DATASET_ID DATASET_TITLE KAGGLE_LICENSE KAGGLE_IS_PRIVATE

mapfile -d '' weight_files < <(find "$WEIGHTS_ROOT" -type f \( -name '*.pth' -o -name '*.pt' \) -print0 | sort -z)

if (( ${#weight_files[@]} == 0 )); then
    echo "No .pth or .pt files found in $WEIGHTS_ROOT" >&2
    exit 1
fi

for src in "${weight_files[@]}"; do
    rel="${src#$WEIGHTS_ROOT/}"
    mkdir -p "$STAGE_DIR/$(dirname "$rel")"
    cp -a "$src" "$STAGE_DIR/$rel"
done

"$PYTHON_BIN" - "$STAGE_DIR/dataset-metadata.json" <<'PY'
import json
import os
import sys

metadata = {
    "title": os.environ["DATASET_TITLE"],
    "id": os.environ["DATASET_ID"],
    "licenses": [{"name": os.environ["KAGGLE_LICENSE"]}],
    "isPrivate": os.environ["KAGGLE_IS_PRIVATE"].lower() == "true",
}

with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
    f.write("\n")
PY

FILE_COUNT="$(find "$STAGE_DIR" -type f ! -name 'dataset-metadata.json' | wc -l | tr -d ' ')"
TOTAL_SIZE="$(du -sh "$STAGE_DIR" | cut -f1)"
echo "Prepared $FILE_COUNT weight files for $DATASET_ID"
echo "Source root: $WEIGHTS_ROOT"
echo "Stage size: $TOTAL_SIZE"
echo "Directory mode: $KAGGLE_DIR_MODE"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run only."
    exit 0
fi

if "${KAGGLE_CMD[@]}" datasets metadata "$DATASET_ID" -p "$META_DIR" >/dev/null 2>&1; then
    "${KAGGLE_CMD[@]}" datasets version -p "$STAGE_DIR" -m "$VERSION_NOTE" -r "$KAGGLE_DIR_MODE"
else
    "${KAGGLE_CMD[@]}" datasets create -p "$STAGE_DIR" -r "$KAGGLE_DIR_MODE"
fi
