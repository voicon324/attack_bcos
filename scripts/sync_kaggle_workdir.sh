#!/bin/bash

# Usage: ./sync_kaggle_workdir.sh [push|pull]
# Default is push (local -> remote)
# Sync current directory to /kaggle/working on the remote machine (or vice versa)
# Excludes .git and other potentially large/unnecessary directories

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DIRECTION="${1:-push}"
SSH_OPT="ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -i ~/.ssh/kaggle_rsa -p 9191"
REMOTE="root@127.0.0.1:/kaggle/working"

# Common excludes
EXCLUDES=(
    --exclude '.git'
    --exclude 'artifacts'
    --exclude '__pycache__'
    --exclude '*.pyc'
    --exclude '.ipynb_checkpoints'
    --exclude 'sync_kaggle_workdir.sh'
    --exclude '*.ipynb'
    --exclude 'venv'
    --exclude '*.pt'
    --exclude '*.pth'
    --exclude 'checkpoint*'
    # --exclude '*.png' 
)

if [ "$DIRECTION" == "pull" ]; then
    echo "Pulling from Kaggle..."
    # Note: trailing slash on remote path is important to sync contents
    rsync -avz "${EXCLUDES[@]}" -e "$SSH_OPT" "$REMOTE/" .
else
    echo "Pushing to Kaggle..."
    rsync -avz "${EXCLUDES[@]}" -e "$SSH_OPT" . "$REMOTE"
fi
