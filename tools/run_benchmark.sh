#!/usr/bin/env bash
set -euo pipefail
# ───── static cluster paths ────────────────────────────────────────────
ROOT=/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca
TRAINING=$ROOT/TRAINING
BENCHMARK=$ROOT/BENCHMARK
SRC=$ROOT/ANIMAL-SPOT
ENV=/user/d.arizaecheverri/u17184/.project/dir.project/micromamba/envs/animal-spot
# ----------------------------------------------------------------------

if [[ $# -ne 2 ]]; then
  echo "Usage: run_benchmark.sh <BENCH-CORPUS> <VARIANTS.json>"
  exit 1
fi

CORPUS=$1          # e.g. $BENCHMARK/benchmark_corpus_v1
VARJSON=$2         # json with seq_len/hop/threshold variants

# ─── 1. generate cfgs + batch scripts ─────────────────────────────────
python "$ROOT/tools/benchmark_factory.py" \
       --corpus-root "$CORPUS" \
       --variants-json "$VARJSON"

# ─── 2. submit the prediction arrays (one per model) ──────────────────
PRED_MASTER="$BENCHMARK/jobs/pred_models.batch"
sbatch "$PRED_MASTER"
MASTER_JOB=$(squeue -u "$USER" -h -n pred_models.batch -o "%i")

echo "⏳ waiting for prediction arrays to finish (master job $MASTER_JOB)…"
while squeue -h -j "$MASTER_JOB" &>/dev/null; do
   sleep 60
done
echo "✅  all prediction jobs done"

# ─── 3. run evaluation only for *completed* outputs ───────────────────
find "$BENCHMARK/runs" -type f -name '*_predict_output.log' |
while read -r logf; do
   outdir=$(dirname "$logf")
   cfg=$(echo "$outdir" | sed 's#/runs/#/cfg/#')/eval.cfg
   if [[ ! -f "$outdir"/evaluation.done ]]; then
       python "$SRC/EVALUATION/start_evaluation.py" "$cfg" && touch "$outdir"/evaluation.done
   fi
done

echo "🎉 benchmark pipeline finished — evaluations stored next to each run"
