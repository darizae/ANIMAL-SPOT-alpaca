#!/bin/bash
###############################################################################
#  ANIMAL-SPOT alpaca project – evaluation launcher + index builder
#  Evaluates all variants found in BENCHMARK/cfg without filtering.
#
#  Usage:
#    bash tools/run_evaluation.sh
###############################################################################

set -euo pipefail

# ─── repo-localised anchors + .env ───────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/.env"
  set +a
fi

BENCHMARK_ROOT="${BENCHMARK_ROOT:-${REPO_ROOT}/BENCHMARK}"
EVAL_SCRIPT="${REPO_ROOT}/EVALUATION/start_evaluation.py"
PRED_IDX_SCRIPT="${REPO_ROOT}/tools/build_pred_index.py"

# ─── micromamba env ──────────────────────────────────────────────────────────
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-}"
if [[ -n "${MAMBA_EXE:-}" ]]; then
  eval "$(${MAMBA_EXE} shell hook --shell=bash)"
else
  # fallback to PATH-resolved micromamba
  eval "$(micromamba shell hook --shell=bash)"
fi
micromamba activate "${MAMBA_ENV_NAME:-animal-spot}"

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
