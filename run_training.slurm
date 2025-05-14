#!/bin/bash
#SBATCH --job-name=animalspot_train
#SBATCH --partition=kisski
#SBATCH --gpus=1
#SBATCH --constraint=80gb_vram
#SBATCH --time=04:00:00
#SBATCH --account=kisski-dpz-alpaca-hum
#SBATCH --output=/mnt/vast-kisski/projects/kisski-dpz-alpaca-hum/logs/%x-%j.out
#SBATCH --chdir=/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca

# Load micromamba + env
module load micromamba
eval "$(micromamba shell hook --shell=bash)"
micromamba activate /mnt/vast-kisski/projects/kisski-dpz-alpaca-hum/envs/animal-spot

# Run the training script
python ./ANIMAL-SPOT/TRAINING/start_training.py TRAINING/monkparakeet_server.cfg
