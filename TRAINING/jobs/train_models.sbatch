#!/bin/bash
#SBATCH --job-name=animalspot_train
#SBATCH --partition=kisski
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=A100:2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --account=kisski-alpaca-2
#SBATCH --output=/mnt/vast-kisski/projects/kisski-alpaca-2/repos/ANIMAL-SPOT-alpaca/TRAINING/runs/job_logs/%x-%A_%a.out
#SBATCH --error=/mnt/vast-kisski/projects/kisski-alpaca-2/repos/ANIMAL-SPOT-alpaca/TRAINING/runs/job_logs/%x-%A_%a.err
#SBATCH -a 0-0%1
#SBATCH --chdir=/mnt/vast-kisski/projects/kisski-alpaca-2/repos/ANIMAL-SPOT-alpaca

set -euo pipefail

# --- repo & env bootstrap (no mamba) ---
export REPO_ROOT="/mnt/vast-kisski/projects/kisski-alpaca-2/repos/ANIMAL-SPOT-alpaca"

# Load project .env so REPO_ROOT/DATA_ROOT/TRAINING_ROOT/etc. are available
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

# Activate repo-local venv
VENV_PY="$REPO_ROOT/.venv/bin/python"
VENV_ACT="$REPO_ROOT/.venv/bin/activate"
if [[ ! -x "$VENV_PY" ]]; then
  echo "❌ Missing venv at $REPO_ROOT/.venv."
  echo "   On a login node, run:"
  echo "     cd $REPO_ROOT && python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi
source "$VENV_ACT"
PY="$VENV_PY"

# Sanity echo
echo "Using python: $(which python)"
python -c "import sys; print('sys.version:', sys.version)"
python -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| tv', torchvision.__version__)"

CONFIGS=(/mnt/vast-kisski/projects/kisski-alpaca-2/repos/ANIMAL-SPOT-alpaca/TRAINING/cfg/v1_clipwise_balanced_nph_2/alpaca_server.cfg)
"$PY" "/mnt/vast-kisski/projects/kisski-alpaca-2/repos/ANIMAL-SPOT-alpaca/TRAINING/start_training.py" "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
