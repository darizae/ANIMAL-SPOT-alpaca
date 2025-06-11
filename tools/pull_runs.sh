#!/usr/bin/env bash
# pull_results.sh  –  sync artefacts from GWDG-HPC to local repo
#
# Usage
#   ./pull_results.sh                 # sync everything
#   ./pull_results.sh v3_tape_prop    # sync only that sub-tree
#
# Notes
# ─────
# • TRAINING → grabs */summaries/**
# • BENCHMARK → grabs */evaluation/** and any   index.json   inside that folder
# • Variant argument is treated as a *path suffix* under each remote base:
#       └── TRAINING/runs/models/<VARIANT>
#       └── BENCHMARK/runs/<VARIANT>
#   so you can pass e.g.  v3_tape_proportional/len500_hop050_th10
#

set -euo pipefail

################################################################################
# CONFIG
################################################################################
REMOTE_USER="u17184"
REMOTE_HOST="glogin-gpu.hpc.gwdg.de"
REMOTE_REPO="/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca"
LOCAL_REPO="$HOME/repos/ANIMAL-SPOT-alpaca"

# bases we want to mirror
REMOTE_TRAIN_BASE="${REMOTE_REPO}/TRAINING/runs/models"
LOCAL_TRAIN_BASE="${LOCAL_REPO}/TRAINING/runs/models"

REMOTE_BENCH_BASE="${REMOTE_REPO}/BENCHMARK/runs"
LOCAL_BENCH_BASE="${LOCAL_REPO}/BENCHMARK/runs"

# optional sub-path
VARIANT="${1-}"          # empty → sync all

# choose a progress flag that works on macOS *and* GNU
if rsync --version 2>&1 | grep -q 'version 3'; then
  PROG_FLAG="--info=progress2"
else
  PROG_FLAG="--progress"
fi

################################################################################
# 1️⃣  TRAINING  –  summaries
################################################################################
if [[ -n "$VARIANT" ]]; then
  REM_TRAIN_PATH="${REMOTE_TRAIN_BASE}/${VARIANT}"
  LOC_TRAIN_PATH="${LOCAL_TRAIN_BASE}/${VARIANT}"
else
  REM_TRAIN_PATH="${REMOTE_TRAIN_BASE}"
  LOC_TRAIN_PATH="${LOCAL_TRAIN_BASE}"
fi

echo "🟣 TRAINING  ↘  ${REMOTE_USER}@${REMOTE_HOST}:${REM_TRAIN_PATH}"
echo "              →  ${LOC_TRAIN_PATH}"
echo
mkdir -p "${LOC_TRAIN_PATH}"

rsync -avz ${PROG_FLAG} \
  --include '*/' \
  --include 'summaries/***' \
  --exclude '*' \
  "${REMOTE_USER}@${REMOTE_HOST}:${REM_TRAIN_PATH}/" \
  "${LOC_TRAIN_PATH}/"

################################################################################
# 2️⃣  BENCHMARK  –  evaluation outputs
################################################################################
if [[ -n "$VARIANT" ]]; then
  REM_BENCH_PATH="${REMOTE_BENCH_BASE}/${VARIANT}"
  LOC_BENCH_PATH="${LOCAL_BENCH_BASE}/${VARIANT}"
else
  REM_BENCH_PATH="${REMOTE_BENCH_BASE}"
  LOC_BENCH_PATH="${LOCAL_BENCH_BASE}"
fi

echo
echo "🟢 BENCHMARK ↘  ${REMOTE_USER}@${REMOTE_HOST}:${REM_BENCH_PATH}"
echo "              →  ${LOC_BENCH_PATH}"
echo
mkdir -p "${LOC_BENCH_PATH}"

rsync -avz ${PROG_FLAG} \
  --include '*/' \
  --include 'evaluation/***' \
  --include 'evaluation/index.json' \
  --exclude '*' \
  "${REMOTE_USER}@${REMOTE_HOST}:${REM_BENCH_PATH}/" \
  "${LOC_BENCH_PATH}/"

echo
echo "✅  Sync complete"
