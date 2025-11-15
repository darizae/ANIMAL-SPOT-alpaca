#!/bin/bash
set -euo pipefail

REPO_ROOT=/projects/extern/kisski/kisski-alpaca-2/dir.project/repos/ANIMAL-SPOT-alpaca
TRAINING_ROOT="$REPO_ROOT/TRAINING/runs"
BENCHMARK_ROOT="$REPO_ROOT/BENCHMARK"

export REPO_ROOT TRAINING_ROOT BENCHMARK_ROOT

cd "$REPO_ROOT"

DIRS=(
  388_m32_20250213
  401_m28_20250213
  408_m25_20250213
  412_m9_20250213
  4212_m43_20250213_33files
  5495_m20_20250213
  6976_m37_20250213
  7166_m48_20250213
  7919_m41_20230213
  7935_m15_20250213
  7991_m31_20250213
  8117_m42_20250213
  8161_m19_20250213
  PINK_m16_20250213
  TINA_m44_20250213_41files
)

for DIR in "${DIRS[@]}"; do
  echo "=== Building configs and submitting prediction for $DIR ==="
  PREDICT_IN="$REPO_ROOT/data/$DIR" make benchmark-configs
  make gpu-predict
done
