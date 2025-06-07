#!/usr/bin/env bash
# pull_runs.sh – sync TRAINING/runs/models/*/summaries from HPC to local repo
# Usage:
#   ./pull_runs.sh              # all variants
#   ./pull_runs.sh v2_quality   # just one

set -euo pipefail

# ─── CONFIG ──────────────────────────────────────────────────────────────────
REMOTE_USER="u17184"
REMOTE_HOST="glogin-gpu.hpc.gwdg.de"
REMOTE_BASE="/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/TRAINING/runs/models"
LOCAL_BASE="$HOME/repos/ANIMAL-SPOT-alpaca/TRAINING/runs/models"

VARIANT="${1-}"       # empty string means “all variants”

# ─── rsync binary & progress flag (macOS vs GNU) ────────────────────────────
RSYNC_BIN=rsync
if ${RSYNC_BIN} --version 2>&1 | grep -q 'version 3'; then
    PROG_FLAG="--info=progress2"
else
    PROG_FLAG="--progress"
fi

# ─── build remote/local paths ───────────────────────────────────────────────
if [[ -n "$VARIANT" ]]; then
    REMOTE_PATH="${REMOTE_BASE}/${VARIANT}"
    LOCAL_PATH="${LOCAL_BASE}/${VARIANT}"
else
    REMOTE_PATH="${REMOTE_BASE}"
    LOCAL_PATH="${LOCAL_BASE}"
fi

echo "↘  syncing summaries from ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
echo "   to   ${LOCAL_PATH}"
echo

mkdir -p "${LOCAL_PATH}"

# ─── rsync: include only */summaries/** and the dirs that lead there ────────
${RSYNC_BIN} -avz ${PROG_FLAG} \
  --include '*/' \
  --include 'summaries/***' \
  --exclude '*' \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/" \
  "${LOCAL_PATH}/"

echo
echo "✅  Summaries sync complete"
