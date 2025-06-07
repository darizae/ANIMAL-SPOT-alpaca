#!/bin/bash
#SBATCH --job-name=animalspot_train
#SBATCH --partition=kisski
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=A100:2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --account=kisski-dpz-alpaca-hum
#SBATCH --output=/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/TRAINING/runs/job_logs/%x-%A_%a.out
#SBATCH --error=/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/TRAINING/runs/job_logs/%x-%A_%a.err
#SBATCH -a 0-2%3
#SBATCH --chdir=/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca

# ---- module / env bootstrap ----
export PATH=/user/d.arizaecheverri/u17184/.project/dir.project/micromamba:$PATH
eval "$(micromamba shell hook --shell=bash)"
micromamba activate /user/d.arizaecheverri/u17184/.project/dir.project/micromamba/envs/animal-spot

CONFIGS=(/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/TRAINING/cfg/v1_random/alpaca_server.cfg /user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/TRAINING/cfg/v2_quality/alpaca_server.cfg /user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/TRAINING/cfg/v3_tape_proportional/alpaca_server.cfg)
python /user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/TRAINING/start_training.py "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
