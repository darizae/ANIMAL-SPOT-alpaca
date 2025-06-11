#!/bin/bash
# Run this from the root of the repo: ANIMAL-SPOT-alpaca/

cd "$(dirname "$0")/.." || exit 1

echo "[INFO] Starting evaluation pass..."

LOG_DIR="ANIMAL-SPOT/BENCHMARK/runs"

# TEMPORARY: Enable to only evaluate a specific subset of variants
TEST_ONLY=true

# Define whitelist of variants to include in test mode
WHITELIST=("len500_hop050_th60" "len500_hop050_th70")

find "$LOG_DIR" -type f -name '*_predict_output.log' |
while read -r logf; do
  outdir=$(dirname "$logf")

  # If test mode is enabled, skip entries not in whitelist
  if [ "$TEST_ONLY" = true ]; then
    match=false
    for allowed in "${WHITELIST[@]}"; do
      if [[ "$outdir" == *"$allowed"* ]]; then
        match=true
        break
      fi
    done
    if [ "$match" = false ]; then
      echo "[SKIP] Not in whitelist: $outdir"
      continue
    fi
  fi

  cfg=$(echo "$outdir" | sed 's#/runs/#/cfg/#' | sed 's#/prediction##')/eval.cfg
  done_flag="$outdir/evaluation.done"

  if [[ -f "$done_flag" ]]; then
    echo "[SKIP] Already evaluated: $cfg"
    continue
  fi

  echo "[EVAL] Running evaluation for $cfg"
  python ANIMAL-SPOT/EVALUATION/start_evaluation.py "$cfg" && touch "$done_flag"
done

echo "[INFO] All evaluations completed."
