#!/bin/bash
#SBATCH --job-name=animalspot_train
#SBATCH --partition=kisski
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=A100:2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --account=kisski-dpz-alpaca-hum
#SBATCH --output=/mnt/vast-kisski/projects/kisski-alpaca-2/repos/ANIMAL-SPOT-alpaca/TRAINING/runs/job_logs/%x-%A_%a.out
#SBATCH --error=/mnt/vast-kisski/projects/kisski-alpaca-2/repos/ANIMAL-SPOT-alpaca/TRAINING/runs/job_logs/%x-%A_%a.err
#SBATCH -a 0-0%1
#SBATCH --chdir=/mnt/vast-kisski/projects/kisski-alpaca-2/repos/ANIMAL-SPOT-alpaca

export MAMBA_ROOT_PREFIX=/projects/extern/kisski/kisski-alpaca-2/dir.project/mamba
eval "$(/projects/extern/kisski/kisski-alpaca-2/dir.project/tools/bin/micromamba shell hook --shell=bash)"
micromamba activate animal-spot

CONFIGS=(/mnt/vast-kisski/projects/kisski-alpaca-2/repos/ANIMAL-SPOT-alpaca/TRAINING/cfg/v1_clipwise_balanced_nph_2/alpaca_server.cfg)
python /mnt/vast-kisski/projects/kisski-alpaca-2/repos/ANIMAL-SPOT-alpaca/TRAINING/start_training.py "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
