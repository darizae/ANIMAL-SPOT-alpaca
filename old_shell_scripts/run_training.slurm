#!/bin/bash
#SBATCH --job-name=animalspot_train
#SBATCH --partition=kisski
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=A100:2
#SBATCH --cpus-per-task=8
#SBATCH --constraint=80gb_vram
#SBATCH --time=00:45:00
#SBATCH --account=kisski-dpz-alpaca-hum
#SBATCH --output=/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/job_logs/%x-%j.out
#SBATCH --error=/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/job_logs/%x-%j.err
#SBATCH --chdir=/user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca

# Load micromamba + env
export PATH=/user/d.arizaecheverri/u17184/.project/dir.project/micromamba:$PATH
eval "$(micromamba shell hook --shell=bash)"
micromamba activate /user/d.arizaecheverri/u17184/.project/dir.project/micromamba/envs/animal-spot

# Run the training script
python /user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/TRAINING/start_training.py /user/d.arizaecheverri/u17184/repos/ANIMAL-SPOT-alpaca/TRAINING/cfg/train_seq_len_300/alpaca_server.cfg
