#!/usr/bin/env bash
# pull_runs.sh  – sync TRAINING/runs/* from the HPC server
# Usage:
#   ./pull_runs.sh                # sync everything
#   ./pull_runs.sh <variant>      # sync only TRAINING/runs/models/<variant>

set -euo pipefail

# ─── CONFIG ────────────────────────────────────────────────────────────────────
REMOTE_USER="u17184"
REMOTE_HOST="glogin-gpu.hpc.gwdg.de"
REMOTE_BASE="/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/TRAINING/runs"

# local destination mirrors the repo layout on your Mac/Linux machine
LOCAL_BASE="$HOME/repos/ANIMAL-SPOT-alpaca/TRAINING/runs"

# optional first argument = specific variant folder (e.g. v1_random)
VARIANT="${1-}"

# build remote + local paths
if [[ -n "$VARIANT" ]]; then
  REMOTE_PATH="${REMOTE_BASE}/models/${VARIANT}"
  LOCAL_PATH="${LOCAL_BASE}/models/${VARIANT}"
else
  REMOTE_PATH="${REMOTE_BASE}"
  LOCAL_PATH="${LOCAL_BASE}"
fi

echo "↘  syncing from ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
echo "   to   ${LOCAL_PATH}"
echo

# ensure local path exists
mkdir -p "${LOCAL_PATH}"

# rsync with compression, resume support and progress bar
rsync -avz --info=progress2 \
      "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/" \
      "${LOCAL_PATH}/"

echo
echo "✅  Sync complete"
