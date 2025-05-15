#!/bin/bash
#SBATCH --job-name=animalspot_train
#SBATCH --partition=kisski
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=A100:4
#SBATCH --cpus-per-task=16
#SBATCH --constraint=80gb_vram
#SBATCH --time=04:00:00
#SBATCH --account=kisski-dpz-alpaca-hum
#SBATCH --output=/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/job_logs/%x-%j.out
#SBATCH --error=/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/job_logs/%x-%j.err
#SBATCH --chdir=/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca

# Load micromamba + env
module load micromamba
eval "$(micromamba shell hook --shell=bash)"
micromamba activate /user/d.arizaecheverri/u17184/micromamba/envs/animal-spot

# Run the training script
python /user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/TRAINING/start_training.py /user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/TRAINING/monkparakeet_server.cfg
