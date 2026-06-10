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
DATASET_SLUG="${KAGGLE_DATASET_SLUG:-attack-bcos-github}"
DATASET_TITLE="${KAGGLE_DATASET_TITLE:-attack-bcos-github}"
DATASET_ID_NO="${KAGGLE_DATASET_ID_NO:-}"
KAGGLE_LICENSE="${KAGGLE_LICENSE:-CC0-1.0}"
KAGGLE_IS_PRIVATE="${KAGGLE_IS_PRIVATE:-true}"
KAGGLE_DIR_MODE="${KAGGLE_DIR_MODE:-zip}"
VERSION_NOTE="${1:-sync $(date '+%Y-%m-%d %H:%M:%S')}"

if [[ ! -f "$KAGGLE_JSON" ]]; then
    echo "Missing $KAGGLE_JSON" >&2
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
export DATASET_ID DATASET_TITLE DATASET_ID_NO KAGGLE_LICENSE KAGGLE_IS_PRIVATE

stage_repo_files() {
    local repo_root="$1"
    local dest_prefix="$2"
    local rel=""

    while IFS= read -r -d '' rel; do
        case "$dest_prefix$rel" in
            kaggle.json|dataset-metadata.json)
                continue
                ;;
            artifacts|artifacts/*)
                continue
                ;;
            imagenet|imagenet/*|weights|weights/*)
                continue
                ;;
            result|result/*|results|results/*|result_*|result_*/*|results_*|results_*/*)
                continue
                ;;
            __pycache__|__pycache__/*|*/__pycache__|*/__pycache__/*|*.pyc|*.pyo)
                continue
                ;;
            scripts/package_kaggle_code_dataset.sh)
                continue
                ;;
            *.zip|*.tar|*.tar.gz|*.tgz|*.7z|*.rar)
                continue
                ;;
        esac

        if [[ -d "$repo_root/$rel" ]] && git -C "$repo_root/$rel" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            stage_repo_files "$repo_root/$rel" "$dest_prefix$rel/"
            continue
        fi

        if [[ ! -e "$repo_root/$rel" && ! -L "$repo_root/$rel" ]]; then
            continue
        fi

        mkdir -p "$STAGE_DIR/$(dirname "$dest_prefix$rel")"
        cp -a "$repo_root/$rel" "$STAGE_DIR/$dest_prefix$rel"
    done < <(git -C "$repo_root" ls-files -z --cached --others --exclude-standard)
}

stage_extra_file() {
    local rel="$1"
    if [[ ! -e "$ROOT/$rel" && ! -L "$ROOT/$rel" ]]; then
        return
    fi
    mkdir -p "$STAGE_DIR/$(dirname "$rel")"
    cp -a "$ROOT/$rel" "$STAGE_DIR/$rel"
}

stage_repo_files "$ROOT" ""
stage_extra_file "data/used_images_500.csv"
stage_extra_file "data/used_images_500_1.csv"
stage_extra_file "data/used_images_500_kaggle.csv"
stage_extra_file "data/used_images_500_rel.csv"
stage_extra_file "data/used_images_1000.csv"

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
if os.environ.get("DATASET_ID_NO"):
    metadata["id_no"] = os.environ["DATASET_ID_NO"]

with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
    f.write("\n")
PY

FILE_COUNT="$(find "$STAGE_DIR" -type f ! -name 'dataset-metadata.json' | wc -l | tr -d ' ')"
TOTAL_SIZE="$(du -sh "$STAGE_DIR" | cut -f1)"
echo "Prepared $FILE_COUNT files for $DATASET_ID"
echo "Stage size: $TOTAL_SIZE"
echo "Directory mode: $KAGGLE_DIR_MODE"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run only."
    exit 0
fi

run_kaggle_write() {
    local log_file
    log_file="$(mktemp)"
    local rc=0
    "${KAGGLE_CMD[@]}" "$@" 2>&1 | tee "$log_file" || rc=$?
    if grep -qE 'Dataset (version )?creation error:' "$log_file"; then
        rc=1
    fi
    rm -f "$log_file"
    return "$rc"
}

if "${KAGGLE_CMD[@]}" datasets metadata "$DATASET_ID" -p "$META_DIR" >/dev/null 2>&1; then
    run_kaggle_write datasets version -p "$STAGE_DIR" -m "$VERSION_NOTE" -r "$KAGGLE_DIR_MODE"
else
    run_kaggle_write datasets create -p "$STAGE_DIR" -r "$KAGGLE_DIR_MODE"
fi
