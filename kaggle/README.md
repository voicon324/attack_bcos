# CamoPatch Kaggle Runs

## Generate Jobs

```bash
python scripts/generate_camopatch_kaggle_jobs.py --output kaggle/camopatch_jobs.json
```

This writes 171 CamoPatch B-cos jobs: 7 models, 3 patch sizes, 3 Linf budgets,
and fixed `random`/`bcos_top1`/`gradcam` positions, excluding ViTC gradcam.
Jobs are staged as offline Kaggle scripts with `machine_shape=NvidiaRtxPro6000`
and the `nvidia-nemotron-model-reasoning-challenge` competition source, matching
the Pro 6000 setup in `template__fixed.ipynb`.

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
  --bundle-target-hours 9.5 \
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
  --auto-bundle-target-hours 9.5
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
