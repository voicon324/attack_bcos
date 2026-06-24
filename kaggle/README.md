# CamoPatch, Patch-RS, LaVAN, And Adversarial Patch Kaggle Runs

## Generate Jobs

```bash
python scripts/generate_camopatch_kaggle_jobs.py --output kaggle/camopatch_jobs.json
```

This writes 171 CamoPatch B-cos jobs: 7 models, 3 patch sizes, 3 Linf budgets,
and fixed `random`/`bcos_top1`/`gradcam` positions, excluding ViTC gradcam.
Jobs are staged as offline Kaggle scripts with `machine_shape=NvidiaRtxPro6000`
and the `arc-prize-2026-arc-agi-3` competition source, matching
the Pro 6000 setup in `template__fixed.ipynb`.

Generate the movable-position size-16, `L_inf=64/256` queue:

```bash
python scripts/generate_camopatch_movable_kaggle_jobs.py \
  --output kaggle/camopatch_movable_s16_linf64_jobs.json
```

This writes 14 jobs: the same 7 B-cos models, init positions `random` and
`bcos_top1`, `fixed_position=false`, and no gradcam. The CamoPatch runner still
validates and zips `summary.csv`, `success_events.csv`, `success_by_query.csv`,
and per-image `.npy` files, including `first_success_query` and final patch
coordinates.

Generate Patch-RS patch-only queues with the same transforms, model matrix,
position rules, `L_inf` budgets, Pro 6000 shape, and offline Kaggle sources:

```bash
python scripts/generate_patchrs_kaggle_jobs.py \
  --output kaggle/patchrs_jobs.json

python scripts/generate_patchrs_movable_kaggle_jobs.py \
  --output kaggle/patchrs_movable_s16_linf64_jobs.json
```

Patch-RS jobs use `attack=patchrs`, Sparse-RS `random_squares` patch
initialization by default, and the shared Kaggle runner dispatches them to
`PatchRS/ConPatchRSBatch.py`. Output zips use the same contract as CamoPatch.

Generate LaVAN queues with the same B-cos models, transforms, initial position
rules, `L_inf` budgets, Pro 6000 shape, and offline Kaggle sources:

```bash
python scripts/generate_lavan_kaggle_jobs.py \
  --output kaggle/lavan_jobs.json

python scripts/generate_lavan_movable_kaggle_jobs.py \
  --output kaggle/lavan_movable_s16_linf64_jobs.json
```

LaVAN jobs use `attack=lavan`, random initialization inside the configured
`L_inf` ball, and dispatch to `LaVAN/ConLaVANBatch.py`. The `queries` field is
the white-box optimization iteration/eval count, defaulting to 10000 for the
shared project matrix. Output zips use the same contract as CamoPatch and
Patch-RS.

Generate A-LinCui Adversarial Patch queues with the same B-cos models,
transforms, initial position rules, `L_inf` budgets, Pro 6000 shape, and
offline Kaggle sources:

```bash
python scripts/generate_adversarial_patch_kaggle_jobs.py \
  --output kaggle/adversarialpatch_jobs.json

python scripts/generate_adversarial_patch_kaggle_jobs.py \
  --s16-linf64-only \
  --output kaggle/adversarialpatch_s16_linf64_positions_jobs.json

python scripts/generate_adversarial_patch_movable_kaggle_jobs.py \
  --output kaggle/adversarialpatch_movable_s16_linf64_jobs.json
```

Adversarial Patch jobs use `attack=adversarial_patch`, dispatch to
`AdversarialPatch/ConAdversarialPatchBatch.py`, default to the upstream fixed
target class `859`, and store `target_class`, `targeted_success`, and
`target_probability` in `summary.csv`. The matrix success metric remains
untargeted by default (`final_prediction != true_label`) for comparability with
the existing CamoPatch, Patch-RS, and LaVAN tables. The default
`queries=10000` matches the project matrix.

## Accounts

The main dataset-update account is `artifacts/secrets/kaggle.json`.
Add an auxiliary runner account at:

```text
kaggle_runs/accounts/aux/kaggle.json
```

The scheduler also auto-discovers `artifacts/secrets/kaggle_*.json`; for example,
`artifacts/secrets/kaggle_dora.json` is used as account `dora`.

or pass an accounts file:

```bash
python scripts/run_kaggle_scheduler.py --accounts-config kaggle/accounts.example.json --once --poll-only
```

The movable queue uses `kaggle/accounts_movable_md.json`, which points to
`artifacts/secrets/kaggle_md1.json` through `kaggle_md4.json` and sets
`max_running=2` per account.

Each account defaults to `max_running=2`. Do not commit any `kaggle.json`.
Optional account fields can override the quota estimator:
`weekly_gpu_quota_hours`, `quota_reset_weekday`, `quota_reset_hour`,
`quota_reset_timezone`, `auto_bundle_under_quota_hours`,
`auto_bundle_target_hours`, and `bundle_max_jobs`.

## Submit And Monitor

Smoke submit one job:

```bash
python scripts/run_kaggle_scheduler.py \
  --jobs-config kaggle/camopatch_jobs.json \
  --accounts-config kaggle/accounts.example.json \
  --max-submit 1 \
  --once
```

Full queue:

```bash
python scripts/run_kaggle_scheduler.py \
  --jobs-config kaggle/camopatch_jobs.json \
  --accounts-config kaggle/accounts.example.json
```

Tail bundle mode, for accounts with only a few GPU quota hours left:

```bash
python scripts/run_kaggle_scheduler.py \
  --jobs-config kaggle/camopatch_jobs.json \
  --bundle-target-hours 7.5 \
  --bundle-max-jobs 5
```

The scheduler also has automatic tail bundle mode enabled by default. It
estimates weekly GPU usage from local `state.json`, counting one active Kaggle
kernel per regular job and one active kernel per bundle. By default it assumes a
30h weekly quota, reset on Saturday 00:00 UTC, and switches an account to bundle
submissions when estimated remaining quota is at or below 4h:

```bash
python scripts/run_kaggle_scheduler.py \
  --jobs-config kaggle/camopatch_jobs.json \
  --auto-bundle-under-quota-hours 4 \
  --auto-bundle-target-hours 7.5
```

Set `--auto-bundle-under-quota-hours 0` to disable automatic bundle switching.
The estimate is intentionally conservative and can be wrong if the same Kaggle
account runs GPU notebooks outside this scheduler, or if Kaggle assigns that
account a quota other than the configured value. If the Kaggle UI shows a
different reset time, override `--quota-reset-timezone`, `--quota-reset-hour`,
or the per-account fields in the accounts config.

Bundle mode still records each condition as its own job in `state.json` and on
the dashboard. The scheduler submits one Kaggle kernel containing multiple
queued jobs from the same model, then the Kaggle runner executes them
sequentially and writes one result zip per original job.

Each Kaggle job writes `/kaggle/working/<job_id>_result.zip`. The zip contains
`outputs/summary.csv`, `outputs/success_events.csv`,
`outputs/success_by_query.csv`, per-image `.npy` files, `manifest.json`, and
`run.log`. `summary.csv` has one row per image, including
`first_success_query` plus final patch coordinates
`patch_position_y`, `patch_position_x`, `patch_position_h`, and
`patch_position_w` on the saved adversarial image. Scheduler state is written
under `kaggle_runs/`: `state.json`, `progress.log`, `dashboard.tsv`, staged
kernels, push logs, downloaded outputs, and result zips.

For Kaggle smoke tests only, a job config may set `limit_images` to run the
first N rows from `images_csv`; omit it for the full 1000-image matrix jobs.

## Update Code Dataset

After pushing code to GitHub, update the Kaggle code dataset:

```bash
KAGGLE_JSON=artifacts/secrets/kaggle.json \
KAGGLE_DATASET_SLUG=attack-bcos-github \
scripts/package_kaggle_code_dataset.sh "sync after github push"
```
