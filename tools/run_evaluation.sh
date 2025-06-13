#!/bin/bash
###############################################################################
#  ANIMAL-SPOT alpaca project – evaluation launcher + index builder
#
#  Evaluates all variants found in BENCHMARK/cfg without filtering.
#
#  Usage:
#  ---------------------------------------------------------------------------
#  bash run_evaluation.sh
#
###############################################################################

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

# ─── iterate over every eval.cfg ──────────────────────────────────────────────
find "${BENCHMARK_ROOT}/cfg" -type f -name eval.cfg | sort | while read -r CFG; do
  variant_dir="$(dirname "${CFG}")"              # …/cfg/<model>/<variant>
  model_dir="$(dirname "${variant_dir}")"        # …/cfg/<model>
  model_name="$(basename "${model_dir}")"
  variant_name="$(basename "${variant_dir}")"
  RUN_ROOT="${BENCHMARK_ROOT}/runs/${model_name}/${variant_name}"

  echo "─── Evaluating: ${RUN_ROOT}"

  # 1️⃣ evaluation
  python "${EVAL_SCRIPT}" "${CFG}"

  # 2️⃣ index build
  python "${PRED_IDX_SCRIPT}" "${RUN_ROOT}"
done

echo "✅ Finished all evaluations at $(date)"
