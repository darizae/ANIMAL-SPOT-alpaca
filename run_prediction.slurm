#!/bin/bash
#SBATCH --job-name=animalspot_pred
#SBATCH --partition=kisski
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=A100:1          # prediction is light
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --account=kisski-dpz-alpaca-hum
#SBATCH --output=/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/job_logs/%x-%j.out
#SBATCH --error=/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/job_logs/%x-%j.err
#SBATCH --chdir=/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca

module load micromamba
eval "$(micromamba shell hook --shell=bash)"
micromamba activate /user/d.arizaecheverri/u17184/micromamba/envs/animal-spot

python /user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/PREDICTION/start_prediction.py \
       /user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/PREDICTION/cfg/alpaca_server_pred.cfg
