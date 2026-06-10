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

Scheduler state is written under `kaggle_runs/`: `state.json`,
`progress.log`, `dashboard.tsv`, staged kernels, push logs, downloaded outputs,
and result zips.

## Update Code Dataset

After pushing code to GitHub, update the Kaggle code dataset:

```bash
KAGGLE_JSON=artifacts/secrets/kaggle.json \
KAGGLE_DATASET_SLUG=attack-bcos-github \
scripts/package_kaggle_code_dataset.sh "sync after github push"
```
