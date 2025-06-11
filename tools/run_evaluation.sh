#!/bin/bash
###############################################################################
#  ANIMAL-SPOT alpaca project – evaluation launcher + index builder
#
#  Adds an optional "--filter-complete" (or -f) switch:
#    • If ON, only run evaluation for variants whose prediction step is finished.
#    • “Finished” := exactly 13  *.predict_output.log  files exist in
#         BENCHMARK/runs/<model>/<variant>/prediction/output
#
#  Usage examples
#  ---------------------------------------------------------------------------
#  # vanilla (evaluate everything):
#  bash run_evaluation.sh
#
#  # only evaluate completed prediction runs:
#  bash run_evaluation.sh --filter-complete
#
###############################################################################

# ─── CLI parsing ──────────────────────────────────────────────────────────────
FILTER_COMPLETE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--filter-complete) FILTER_COMPLETE=true; shift ;;
    -h|--help)
      echo "Usage: $0 [--filter-complete|-f]"
      exit 0
      ;;
    *) echo "Unknown flag $1"; exit 1 ;;
  esac
done

# ─── absolute, repo-localised anchors ─────────────────────────────────────────
REPO_ROOT="/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca"
BENCHMARK_ROOT="${REPO_ROOT}/BENCHMARK"
EVAL_SCRIPT="${REPO_ROOT}/EVALUATION/start_evaluation.py"
PRED_IDX_SCRIPT="${REPO_ROOT}/tools/build_pred_index.py"

# ─── micromamba env ──────────────────────────────────────────────────────────
export PATH="/user/d.arizaecheverri/u17184/.project/dir.project/micromamba:${PATH}"
set +u        # deactivate nounset while sourcing activation scripts
eval "$(micromamba shell hook --shell=bash)"
micromamba activate /user/d.arizaecheverri/u17184/.project/dir.project/micromamba/envs/animal-spot
set -u        # re-enable nounset

echo "▶ Starting evaluation at $(date)"
$FILTER_COMPLETE && echo "   Mode : filter-complete (only runs with 13 prediction logs)"

# ─── iterate over every eval.cfg ──────────────────────────────────────────────
find "${BENCHMARK_ROOT}/cfg" -type f -name eval.cfg | sort | while read -r CFG; do
  variant_dir="$(dirname "${CFG}")"              # …/cfg/<model>/<variant>
  model_dir="$(dirname "${variant_dir}")"        # …/cfg/<model>
  model_name="$(basename "${model_dir}")"
  variant_name="$(basename "${variant_dir}")"
  RUN_ROOT="${BENCHMARK_ROOT}/runs/${model_name}/${variant_name}"

  # optional filtering --------------------------------------------------------
  if $FILTER_COMPLETE; then
    PRED_OUT="${RUN_ROOT}/prediction/output"
    n_logs=$(find "${PRED_OUT}" -maxdepth 1 -type f -name '*predict_output.log' 2>/dev/null | wc -l)
    if [[ "$n_logs" -ne 13 ]]; then
      echo "⚠️  Skipping (incomplete) ${RUN_ROOT} — ${n_logs}/13 logs present"
      continue
    fi
  fi

  echo "─── Evaluating: ${RUN_ROOT}"

  # 1️⃣ evaluation
  python "${EVAL_SCRIPT}" "${CFG}"

  # 2️⃣ index build
  python "${PRED_IDX_SCRIPT}" "${RUN_ROOT}"
done


echo "✅ Finished all evaluations at $(date)"
