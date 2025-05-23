#!/bin/bash
#SBATCH --job-name=animalspot_pred
#SBATCH --partition=kisski
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=A100:2
#SBATCH --cpus-per-task=8
#SBATCH --constraint=80gb_vram
#SBATCH --time=02:20:00
#SBATCH --account=kisski-dpz-alpaca-hum
#SBATCH --output=/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/job_logs/%x-%j.out
#SBATCH --error=/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/job_logs/%x-%j.err
#SBATCH --chdir=/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca

export PATH=/user/d.arizaecheverri/u17184/.project/dir.project/micromamba:$PATH
eval "$(micromamba shell hook --shell=bash)"
micromamba activate /user/d.arizaecheverri/u17184/.project/dir.project/micromamba/envs/animal-spot

python /user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/PREDICTION/start_prediction.py \
       /user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/PREDICTION/cfg/alpaca_server_pred.cfg
