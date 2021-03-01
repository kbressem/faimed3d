#!/bin/bash
#SBATCH --job-name=r101_3d
#SBATCH --output=r101_3d.txt
#SBATCH --cpus-per-gpu=15
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem-per-cpu=2000M
#SBATCH --nodes=1
source /data/users/bressekk/work/conda/etc/profile.d/conda.sh
conda activate fastai-v2
python -m fastai.launch /data/users/bressekk/work/faimed3d/pretrained/r101_3d.py
