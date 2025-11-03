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

VENV_ACT="${REPO_ROOT}/.venv/bin/activate"
VENV_PY="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "❌ Missing venv at ${REPO_ROOT}/.venv."
  echo "   On a login node, run:"
  echo "     cd ${REPO_ROOT} && python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

# shellcheck source=/dev/null
source "${VENV_ACT}"
PY="${VENV_PY}"

echo "▶ Starting evaluation at $(date)"
echo "   Using python: $(which python)"
python -c "import sys; print('   sys.version:', sys.version)"

# ─── iterate over every eval.cfg ──────────────────────────────────────────────
find "${BENCHMARK_ROOT}/cfg" -type f -name eval.cfg | sort | while read -r CFG; do
  variant_dir="$(dirname "${CFG}")"              # …/cfg/<model>/<variant>
  model_dir="$(dirname "${variant_dir}")"        # …/cfg/<model>
  model_name="$(basename "${model_dir}")"
  variant_name="$(basename "${variant_dir}")"

  RUN_ROOT="${BENCHMARK_ROOT}/runs/${model_name}/${variant_name}"
  echo "─── Evaluating: ${RUN_ROOT}"

  # 1️⃣ evaluation
  "${PY}" "${EVAL_SCRIPT}" "${CFG}"

  # 2️⃣ index build
  "${PY}" "${PRED_IDX_SCRIPT}" "${RUN_ROOT}"
done

echo "✅ Finished all evaluations at $(date)"
