# Kaggle Runbook for Attack B-cos

Read this file first in any new Codex/chat session before touching Kaggle.

## Goal

Run and monitor the CamoPatch, Patch-RS, LaVAN, and Adversarial Patch B-cos Kaggle matrices from this repo.

- Main fixed-position queue: `kaggle/camopatch_jobs.json`
- Movable-position queue: `kaggle/camopatch_movable_s16_linf64_jobs.json`
- Patch-RS fixed-position queue: `kaggle/patchrs_jobs.json`
- Patch-RS movable-position queue: `kaggle/patchrs_movable_s16_linf64_jobs.json`
- LaVAN fixed-position queue: `kaggle/lavan_jobs.json`
- LaVAN movable-position queue: `kaggle/lavan_movable_s16_linf64_jobs.json`
- Adversarial Patch fixed-position queue: `kaggle/adversarialpatch_jobs.json`
- Adversarial Patch size-16, `L_inf=64/256` queue: `kaggle/adversarialpatch_s16_linf64_positions_jobs.json`
- Adversarial Patch movable-position queue: `kaggle/adversarialpatch_movable_s16_linf64_jobs.json`
- Scheduler: `scripts/run_kaggle_scheduler.py`
- Main fixed run root: `kaggle_runs_success_query_full`
- Movable run root: `kaggle_runs_movable_s16_linf64`
- Realtime dashboard server: `scripts/serve_kaggle_dashboard.py`
- Result aggregation: `scripts/aggregate_camopatch_all_results.py`

Use the `bcos` conda environment:

```bash
cd /home/hkduy/workplace/attack-bcos
conda activate bcos
```

## Non-negotiables

- Never print, stage, or commit `kaggle.json` keys.
- Never put credentials inside a staged Kaggle kernel.
- Use Kaggle datasets for code/weights/images. Kaggle runs have internet off.
- Do not add `git clone`, `pip install` from the internet, or remote downloads to the Kaggle job.
- Use Pro 6000:
  - `machine_shape`: `NvidiaRtxPro6000`
  - `enable_gpu`: `true`
  - `enable_internet`: `false`
- Kaggle source setup must include:
  - code dataset: `hkhnhduy/attack-bcos-github`
  - weights dataset: `hkhnhduy/weights-bcos`
  - ImageNet validation dataset: `sautkin/imagenet1kvalid`
  - competition source: `arc-prize-2026-arc-agi-3`
- Each notebook/script must zip result files to `/kaggle/working/<job_id>_result.zip`.
- Result zips must include per-image first success query and final patch position fields.

## Current Known Local State

These counts came from local `state.json` files when this runbook was written.
Re-check before acting.

```text
kaggle_runs_success_query_full: done=111, queued=60
kaggle_runs_movable_s16_linf64: done=14
kaggle_runs_success_query_smoke2: done=1
kaggle_runs_success_query_smoke: failed=1
kaggle_runs: done=33, failed=38, queued=100
```

The most relevant active roots are:

```text
kaggle_runs_success_query_full
kaggle_runs_movable_s16_linf64
```

Quick re-check:

```bash
python - <<'PY'
import json
from pathlib import Path

for root in [
    "kaggle_runs_success_query_full",
    "kaggle_runs_movable_s16_linf64",
    "kaggle_runs_success_query_smoke2",
    "kaggle_runs_success_query_smoke",
    "kaggle_runs",
]:
    p = Path(root) / "state.json"
    if not p.exists():
        continue
    data = json.loads(p.read_text())
    jobs = data.get("jobs", {})
    counts = {}
    for job in (jobs.values() if isinstance(jobs, dict) else jobs):
        status = job.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    print(root, counts)
PY
```

## Account And Credential Rules

Main dataset-update credential:

```text
artifacts/secrets/kaggle.json
```

Known runner configs:

```text
kaggle/accounts.example.json
kaggle/accounts_movable_md.json
```

`kaggle/accounts.example.json` currently references:

```text
main -> artifacts/secrets/kaggle.json
aux  -> kaggle_runs/accounts/aux/kaggle.json
dora -> artifacts/secrets/kaggle_dora.json
```

`kaggle/accounts_movable_md.json` references:

```text
md1 -> artifacts/secrets/kaggle_md1.json
md2 -> artifacts/secrets/kaggle_md2.json
md3 -> artifacts/secrets/kaggle_md3.json
md4 -> artifacts/secrets/kaggle_md4.json
```

Important: `scripts/run_kaggle_scheduler.py` auto-discovers
`artifacts/secrets/kaggle_*.json` only when `--accounts-config` is not passed.
If `artifacts/secrets/kaggle_family.json` is a multi-account file instead of a
single Kaggle API key with `username` and `key`, do not rely on auto-discovery.
Create an explicit ignored accounts config under `artifacts/secrets/`, or split
each account into:

```text
artifacts/secrets/kaggle_accounts/<account_name>/kaggle.json
```

Safe account check without printing API keys:

```bash
python - <<'PY'
import json
from pathlib import Path

paths = [
    Path("artifacts/secrets/kaggle.json"),
    *sorted(Path("artifacts/secrets").glob("kaggle_*.json")),
    *sorted(Path("artifacts/secrets/kaggle_accounts").glob("*/kaggle.json")),
]
seen = set()
for p in paths:
    if not p.exists() or p.resolve() in seen:
        continue
    seen.add(p.resolve())
    try:
        data = json.loads(p.read_text())
    except Exception as exc:
        print(p, "BAD_JSON", exc)
        continue
    username = data.get("username")
    key_present = bool(data.get("key"))
    print(p, "username=", username or "NOT_SINGLE_KAGGLE_KEY", "key=", key_present)
PY
```

Set credential mode when adding accounts:

```bash
chmod 600 artifacts/secrets/kaggle*.json 2>/dev/null || true
find artifacts/secrets/kaggle_accounts -name kaggle.json -exec chmod 600 {} \; 2>/dev/null || true
find kaggle_runs/accounts -name kaggle.json -exec chmod 600 {} \; 2>/dev/null || true
```

## Regenerate Job Configs

Fixed queue, 171 jobs:

```bash
python scripts/generate_camopatch_kaggle_jobs.py --dry-run
python scripts/generate_camopatch_kaggle_jobs.py \
  --output kaggle/camopatch_jobs.json
```

Expected:

- 7 models
- patch sizes: `16`, `8`, `32` in that priority order
- `L_inf`: `16/256`, `32/256`, `64/256`
- fixed positions: `random`, `bcos_top1`, `gradcam`
- no `gradcam` jobs for `vitc_s` or `vitc_b`
- total fixed jobs: `171`

Movable queue, 14 jobs:

```bash
python scripts/generate_camopatch_movable_kaggle_jobs.py --dry-run
python scripts/generate_camopatch_movable_kaggle_jobs.py \
  --output kaggle/camopatch_movable_s16_linf64_jobs.json
```

Expected:

- models same as fixed queue
- patch size `16`
- `L_inf=64/256`
- init positions: `random`, `bcos_top1`
- `fixed_position=false`
- total movable jobs: `14`

Patch-RS fixed queue, 171 jobs:

```bash
python scripts/generate_patchrs_kaggle_jobs.py --dry-run
python scripts/generate_patchrs_kaggle_jobs.py \
  --output kaggle/patchrs_jobs.json
```

Patch-RS movable queue, 14 jobs:

```bash
python scripts/generate_patchrs_movable_kaggle_jobs.py --dry-run
python scripts/generate_patchrs_movable_kaggle_jobs.py \
  --output kaggle/patchrs_movable_s16_linf64_jobs.json
```

Patch-RS uses the same transforms and position rules as CamoPatch:

- `Resize(224) -> CenterCrop(224) -> ToTensor()`
- fixed: `random`, `bcos_top1`, `gradcam`, with no ViTC `gradcam`
- movable: `random`, `bcos_top1`, `fixed_position=false`
- default patch initialization: Sparse-RS `random_squares`

LaVAN fixed queue, 171 jobs:

```bash
python scripts/generate_lavan_kaggle_jobs.py --dry-run
python scripts/generate_lavan_kaggle_jobs.py \
  --output kaggle/lavan_jobs.json
```

LaVAN movable queue, 14 jobs:

```bash
python scripts/generate_lavan_movable_kaggle_jobs.py --dry-run
python scripts/generate_lavan_movable_kaggle_jobs.py \
  --output kaggle/lavan_movable_s16_linf64_jobs.json
```

LaVAN uses the same transforms and position rules as CamoPatch:

- `Resize(224) -> CenterCrop(224) -> ToTensor()`
- fixed: `random`, `bcos_top1`, `gradcam`, with no ViTC `gradcam`
- movable: `random`, `bcos_top1`, `fixed_position=false`
- default patch initialization: random inside the configured `L_inf` ball
- default `queries=10000`, interpreted as white-box optimization iterations/evals

Adversarial Patch fixed queue, 171 jobs:

```bash
python scripts/generate_adversarial_patch_kaggle_jobs.py --dry-run
python scripts/generate_adversarial_patch_kaggle_jobs.py \
  --output kaggle/adversarialpatch_jobs.json
```

Adversarial Patch size-16, `L_inf=64/256` fixed queue, 19 jobs:

```bash
python scripts/generate_adversarial_patch_kaggle_jobs.py \
  --s16-linf64-only \
  --output kaggle/adversarialpatch_s16_linf64_positions_jobs.json
```

Adversarial Patch movable queue, 14 jobs:

```bash
python scripts/generate_adversarial_patch_movable_kaggle_jobs.py --dry-run
python scripts/generate_adversarial_patch_movable_kaggle_jobs.py \
  --output kaggle/adversarialpatch_movable_s16_linf64_jobs.json
```

Adversarial Patch uses the same transforms and position rules as CamoPatch:

- `Resize(224) -> CenterCrop(224) -> ToTensor()`
- fixed: `random`, `bcos_top1`, `gradcam`, with no ViTC `gradcam`
- movable: `random`, `bcos_top1`, `fixed_position=false`
- default patch initialization: random inside the configured `L_inf` ball
- default `queries=10000`, matching the project matrix
- upstream fixed target class default: `859`
- matrix success remains untargeted by default; summary also stores
  `target_class`, `targeted_success`, and `target_probability`

## Update Code Dataset After Code Changes

If code changes were made, push to GitHub first. Immediately after a successful
push, update the Kaggle code dataset with the main account:

```bash
KAGGLE_JSON=artifacts/secrets/kaggle.json \
KAGGLE_DATASET_SLUG=attack-bcos-github \
scripts/package_kaggle_code_dataset.sh "sync after github push"
```

Do this before submitting more Kaggle jobs. The Kaggle run uses the code dataset,
not `git clone`.

Dry run packaging check:

```bash
KAGGLE_JSON=artifacts/secrets/kaggle.json \
KAGGLE_DATASET_SLUG=attack-bcos-github \
scripts/package_kaggle_code_dataset.sh --dry-run "dry run"
```

## Submit Or Resume Fixed Queue

Use explicit accounts config to avoid accidentally loading malformed family
credential files.

One-cycle status/poll only:

```bash
python scripts/run_kaggle_scheduler.py \
  --jobs-config kaggle/camopatch_jobs.json \
  --accounts-config kaggle/accounts.example.json \
  --run-root kaggle_runs_success_query_full \
  --poll-only \
  --once
```

Resume fixed queue normally:

```bash
python scripts/run_kaggle_scheduler.py \
  --jobs-config kaggle/camopatch_jobs.json \
  --accounts-config kaggle/accounts.example.json \
  --run-root kaggle_runs_success_query_full \
  --poll-interval 300
```

Submit at most one new job, useful for smoke/recovery:

```bash
python scripts/run_kaggle_scheduler.py \
  --jobs-config kaggle/camopatch_jobs.json \
  --accounts-config kaggle/accounts.example.json \
  --run-root kaggle_runs_success_query_full \
  --max-submit 1 \
  --once
```

Tail quota bundle mode:

```bash
python scripts/run_kaggle_scheduler.py \
  --jobs-config kaggle/camopatch_jobs.json \
  --accounts-config kaggle/accounts.example.json \
  --run-root kaggle_runs_success_query_full \
  --auto-bundle-under-quota-hours 4 \
  --auto-bundle-target-hours 7.5 \
  --bundle-max-jobs 5 \
  --poll-interval 300
```

Notes:

- Each account defaults to `max_running=2`.
- Bundle kernels still count as active slots.
- The quota estimator assumes 30 GPU hours per week unless account config says
  otherwise.
- Kaggle quota reset was treated as Saturday in previous work, but always check
  the Kaggle UI if numbers look wrong.

## Submit Or Resume Movable Queue

Movable queue was locally done when this file was written. To poll or resume it:

```bash
python scripts/run_kaggle_scheduler.py \
  --jobs-config kaggle/camopatch_movable_s16_linf64_jobs.json \
  --accounts-config kaggle/accounts_movable_md.json \
  --run-root kaggle_runs_movable_s16_linf64 \
  --poll-interval 300
```

Poll only:

```bash
python scripts/run_kaggle_scheduler.py \
  --jobs-config kaggle/camopatch_movable_s16_linf64_jobs.json \
  --accounts-config kaggle/accounts_movable_md.json \
  --run-root kaggle_runs_movable_s16_linf64 \
  --poll-only \
  --once
```

## Submit Or Resume Patch-RS Queues

Use a separate run root from CamoPatch:

```bash
python scripts/run_kaggle_scheduler.py \
  --jobs-config kaggle/patchrs_jobs.json \
  --accounts-config kaggle/accounts.example.json \
  --run-root kaggle_runs_patchrs_full \
  --poll-interval 300
```

Movable Patch-RS:

```bash
python scripts/run_kaggle_scheduler.py \
  --jobs-config kaggle/patchrs_movable_s16_linf64_jobs.json \
  --accounts-config kaggle/accounts_movable_md.json \
  --run-root kaggle_runs_patchrs_movable_s16_linf64 \
  --poll-interval 300
```

Smoke Patch-RS:

```bash
python scripts/generate_patchrs_kaggle_jobs.py \
  --smoke \
  --output /tmp/patchrs_smoke_jobs.json

python scripts/run_kaggle_scheduler.py \
  --jobs-config /tmp/patchrs_smoke_jobs.json \
  --accounts-config kaggle/accounts.example.json \
  --run-root kaggle_runs_patchrs_smoke \
  --max-submit 1 \
  --once
```

## Submit Or Resume LaVAN Queues

Use separate run roots from CamoPatch and Patch-RS:

```bash
python scripts/run_kaggle_scheduler.py \
  --jobs-config kaggle/lavan_jobs.json \
  --accounts-config kaggle/accounts.example.json \
  --run-root kaggle_runs_lavan_full \
  --poll-interval 300
```

Movable LaVAN:

```bash
python scripts/run_kaggle_scheduler.py \
  --jobs-config kaggle/lavan_movable_s16_linf64_jobs.json \
  --accounts-config kaggle/accounts_movable_md.json \
  --run-root kaggle_runs_lavan_movable_s16_linf64 \
  --poll-interval 300
```

Smoke LaVAN:

```bash
python scripts/generate_lavan_kaggle_jobs.py \
  --smoke \
  --output /tmp/lavan_smoke_jobs.json

python scripts/run_kaggle_scheduler.py \
  --jobs-config /tmp/lavan_smoke_jobs.json \
  --accounts-config kaggle/accounts.example.json \
  --run-root kaggle_runs_lavan_smoke \
  --max-submit 1 \
  --once
```

## Submit Or Resume Adversarial Patch Queues

Use separate run roots from CamoPatch, Patch-RS, and LaVAN:

```bash
python scripts/run_kaggle_scheduler.py \
  --jobs-config kaggle/adversarialpatch_jobs.json \
  --accounts-config kaggle/accounts.example.json \
  --run-root kaggle_runs_adversarialpatch_full \
  --poll-interval 300
```

Size-16, `L_inf=64/256` fixed-position Adversarial Patch:

```bash
python scripts/run_kaggle_scheduler.py \
  --jobs-config kaggle/adversarialpatch_s16_linf64_positions_jobs.json \
  --accounts-config kaggle/accounts_patchrs_all.json \
  --run-root kaggle_runs_adversarialpatch_s16_linf64_positions \
  --auto-bundle-under-quota-hours 0 \
  --poll-interval 300
```

Movable Adversarial Patch:

```bash
python scripts/run_kaggle_scheduler.py \
  --jobs-config kaggle/adversarialpatch_movable_s16_linf64_jobs.json \
  --accounts-config kaggle/accounts_movable_md.json \
  --run-root kaggle_runs_adversarialpatch_movable_s16_linf64 \
  --poll-interval 300
```

Smoke Adversarial Patch:

```bash
python scripts/generate_adversarial_patch_kaggle_jobs.py \
  --smoke \
  --output /tmp/adversarialpatch_smoke_jobs.json

python scripts/run_kaggle_scheduler.py \
  --jobs-config /tmp/adversarialpatch_smoke_jobs.json \
  --accounts-config kaggle/accounts.example.json \
  --run-root kaggle_runs_adversarialpatch_smoke \
  --max-submit 1 \
  --once
```

## Smoke Test

Generate a one-job smoke config:

```bash
python scripts/generate_camopatch_kaggle_jobs.py \
  --smoke \
  --output /tmp/camopatch_smoke_jobs.json
```

Submit one smoke job:

```bash
python scripts/run_kaggle_scheduler.py \
  --jobs-config /tmp/camopatch_smoke_jobs.json \
  --accounts-config kaggle/accounts.example.json \
  --run-root kaggle_runs_success_query_smoke2 \
  --max-submit 1 \
  --once
```

Then poll until done:

```bash
python scripts/run_kaggle_scheduler.py \
  --jobs-config /tmp/camopatch_smoke_jobs.json \
  --accounts-config kaggle/accounts.example.json \
  --run-root kaggle_runs_success_query_smoke2 \
  --poll-only
```

## Monitor Progress

Local files:

```text
<run-root>/state.json
<run-root>/progress.log
<run-root>/dashboard.tsv
<run-root>/jobs/<job_id>/url.txt
<run-root>/jobs/<job_id>/push.log
<run-root>/jobs/<job_id>/output.log
<run-root>/jobs/<job_id>/output/
```

Terminal:

```bash
tail -f kaggle_runs_success_query_full/progress.log
column -t -s $'\t' kaggle_runs_success_query_full/dashboard.tsv | less -S
```

Realtime HTML dashboard on LAN:

```bash
python scripts/serve_kaggle_dashboard.py \
  --run-root kaggle_runs_success_query_full \
  --host 0.0.0.0 \
  --port 8765
```

Find the LAN IP:

```bash
hostname -I
```

Open from another machine on the same network:

```text
http://<LAN_IP>:8765
```

If port `8765` is busy, the server has `--port-attempts`; or pass another port.

## Output Contract

Each Kaggle job should produce:

```text
/kaggle/working/<job_id>_result.zip
```

The zip should contain:

```text
outputs/summary.csv
outputs/success_events.csv
outputs/success_by_query.csv
*.npy per-image output arrays
manifest.json
run.log
```

Required result metadata:

- `first_success_query` for each image
- final patch coordinates on the saved adversarial image:
  - `patch_position_y`
  - `patch_position_x`
  - `patch_position_h`
  - `patch_position_w`
- clean prediction/correctness fields for clean-only analysis

The scheduler marks a job done only when the expected result zip exists and
passes `ZipFile.testzip()`.

## Aggregate Results

Aggregate the downloaded fixed and movable result zips into one analysis folder:

```bash
python scripts/aggregate_camopatch_all_results.py \
  --output-dir artifacts/analysis/camopatch_7models_latest \
  --clean-predictions-csv artifacts/analysis/camopatch_7models_latest/clean_predictions_1000.csv \
  --validate-zips \
  --include-run fixed kaggle_runs_success_query_full kaggle/camopatch_jobs.json \
  --include-run movable kaggle_runs_movable_s16_linf64 kaggle/camopatch_movable_s16_linf64_jobs.json
```

Expected important outputs:

```text
artifacts/analysis/camopatch_7models_latest/summary_all_images.csv
artifacts/analysis/camopatch_7models_latest/summary_clean_correct.csv
artifacts/analysis/camopatch_7models_latest/success_by_query_all_images.csv
artifacts/analysis/camopatch_7models_latest/success_by_query_clean_correct.csv
artifacts/analysis/camopatch_7models_latest/combined_image_results_all.csv
artifacts/analysis/camopatch_7models_latest/combined_image_results_clean_correct.csv
artifacts/analysis/camopatch_7models_latest/included_jobs.csv
artifacts/analysis/camopatch_7models_latest/skipped_jobs.csv
artifacts/analysis/camopatch_7models_latest/manifest.json
```

Zip analysis folder if needed:

```bash
cd artifacts/analysis
zip -r camopatch_7models_latest.zip camopatch_7models_latest
cd -
```

## Generate Charts

Grouped bar charts, one image per config:

```bash
python scripts/plot_camopatch_position_bar_images.py \
  --summary-dir artifacts/analysis/camopatch_7models_latest \
  --output-dir artifacts/analysis/camopatch_7models_latest/charts_position_bars \
  --denominators all clean \
  --format png
```

Fixed query curves:

```bash
python scripts/plot_camopatch_movable_query_curves.py \
  --summary-dir artifacts/analysis/camopatch_7models_latest \
  --output-dir artifacts/analysis/camopatch_7models_latest/charts_fixed_query_curves \
  --position-mode fixed \
  --denominators all clean \
  --query-scale symlog \
  --curve-style smooth \
  --format png
```

Movable query curves:

```bash
python scripts/plot_camopatch_movable_query_curves.py \
  --summary-dir artifacts/analysis/camopatch_7models_latest \
  --output-dir artifacts/analysis/camopatch_7models_latest/charts_movable_query_curves \
  --position-mode movable \
  --denominators all clean \
  --query-scale symlog \
  --curve-style smooth \
  --format png
```

## Google Sheet / Reporting

Target sheet used before:

```text
https://docs.google.com/spreadsheets/d/1txNZGc1CphkDzU6_vomz6RclONfYrzViOiLmA-_8s1M/edit?usp=sharing
```

Previous readable tab layout was generated locally under:

```text
artifacts/analysis/google_sheet_upload/
```

If that folder exists, the latest TSV to paste is usually:

```text
artifacts/analysis/google_sheet_upload/camopatch_results_readable_dashboard.tsv
```

Spreadsheet convention:

- blank cell means config is not finished or not run
- do not write `N/A`
- matrix cells should be compact percentages
- counts can live in best-result/source tables

The previous session used browser automation because the local Google token did
not have Sheets API scope.

## Common Failures And Fixes

### Wrong GPU

Check generated job config:

```bash
python - <<'PY'
import json
from pathlib import Path

for p in [
    "kaggle/camopatch_jobs.json",
    "kaggle/camopatch_movable_s16_linf64_jobs.json",
    "kaggle/patchrs_jobs.json",
    "kaggle/patchrs_movable_s16_linf64_jobs.json",
]:
    data = json.loads(Path(p).read_text())
    shapes = sorted({j.get("machine_shape") for j in data["jobs"]})
    internet = sorted({j.get("enable_internet") for j in data["jobs"]})
    print(p, "machine_shape=", shapes, "enable_internet=", internet)
PY
```

Expected: `NvidiaRtxPro6000` and `False`.

### Competition rules not accepted

If push fails with rules/terms errors, log into that Kaggle account in the
browser and accept the competition rules for:

```text
https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/overview
```

Then rerun the scheduler. Keep the job artifacts; do not delete state blindly.

### Account has running 0

Check:

```bash
python scripts/run_kaggle_scheduler.py \
  --jobs-config kaggle/camopatch_jobs.json \
  --accounts-config kaggle/accounts.example.json \
  --run-root kaggle_runs_success_query_full \
  --poll-only \
  --once
```

Then inspect:

```bash
tail -n 80 kaggle_runs_success_query_full/progress.log
column -t -s $'\t' kaggle_runs_success_query_full/dashboard.tsv | less -S
```

Likely causes:

- account cooldown after no GPU capacity
- weekly GPU quota exhausted
- competition rules not accepted
- account missing from explicit accounts config
- malformed family credential file being auto-discovered

### Need to rerun a failed or stale job

Do not rerun done jobs unless intended. To requeue selected jobs, edit
`state.json` carefully: set `status` to `queued` and clear `account`, `url`,
`slug`, `failure_reason`, and bundle fields. Keep old logs.

Use a small Python script and review the diff afterward.

### Timeout

Timeout means the scheduler did not see a valid expected zip before
`timeout_minutes`. Inspect:

```text
<run-root>/jobs/<job_id>/output.log
<run-root>/jobs/<job_id>/output/
<run-root>/jobs/<job_id>/push.log
```

If the Kaggle output zip exists under a nested folder, the scheduler should find
it recursively. If not, check `expected_zip` in the job config.

## Quick Sanity Commands

Compile edited scripts:

```bash
python -m py_compile \
  PatchRS/ConPatchRSBatch.py \
  kaggle/camopatch_job/run_camopatch_job.py \
  scripts/run_kaggle_scheduler.py \
  scripts/generate_camopatch_kaggle_jobs.py \
  scripts/generate_camopatch_movable_kaggle_jobs.py \
  scripts/generate_patchrs_kaggle_jobs.py \
  scripts/generate_patchrs_movable_kaggle_jobs.py \
  scripts/aggregate_camopatch_all_results.py \
  scripts/plot_camopatch_position_bar_images.py \
  scripts/plot_camopatch_movable_query_curves.py
```

Check job counts and queue priority:

```bash
python - <<'PY'
import json
from pathlib import Path

fixed = json.loads(Path("kaggle/camopatch_jobs.json").read_text())["jobs"]
movable = json.loads(Path("kaggle/camopatch_movable_s16_linf64_jobs.json").read_text())["jobs"]
patchrs_fixed = json.loads(Path("kaggle/patchrs_jobs.json").read_text())["jobs"]
patchrs_movable = json.loads(Path("kaggle/patchrs_movable_s16_linf64_jobs.json").read_text())["jobs"]
print("fixed_jobs", len(fixed))
print("movable_jobs", len(movable))
print("patchrs_fixed_jobs", len(patchrs_fixed))
print("patchrs_movable_jobs", len(patchrs_movable))
print("first_fixed_job", fixed[0]["job_id"], fixed[0]["job_config"])
bad = [
    j["job_id"] for j in fixed + patchrs_fixed
    if j["job_config"]["position"] == "gradcam"
    and j["job_config"]["model"] in {"vitc_s", "vitc_b"}
]
print("bad_vitc_gradcam_jobs", bad)
PY
```

Expected:

```text
fixed_jobs 171
movable_jobs 14
patchrs_fixed_jobs 171
patchrs_movable_jobs 14
first fixed job uses patch_size 16
bad_vitc_gradcam_jobs []
```

## If Unsure

Start with poll-only, not submit:

```bash
python scripts/run_kaggle_scheduler.py \
  --jobs-config kaggle/camopatch_jobs.json \
  --accounts-config kaggle/accounts.example.json \
  --run-root kaggle_runs_success_query_full \
  --poll-only \
  --once
```

Then read `dashboard.tsv`, `progress.log`, and this file again before changing
state or submitting more jobs.
