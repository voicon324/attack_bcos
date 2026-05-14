#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-hcmus:attack_bcos}"
MOUNT_POINT="${MOUNT_POINT:-./remote-project}"

mkdir -p "$MOUNT_POINT"

echo "Mounting $REMOTE -> $MOUNT_POINT"
sshfs "$REMOTE" "$MOUNT_POINT" \
  -o reconnect \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3

echo "Mounted. Unmount with: fusermount -u $MOUNT_POINT"
CmK26pFG