#!/bin/bash
#SBATCH --job-name=ucf-101
#SBATCH --output=ucf-101.txt
#SBATCH --cpus-per-gpu=15
#SBATCH --gres=gpu:tesla:2
#SBATCH --mem-per-cpu=2000M
#SBATCH --nodes=1
source /data/users/bressekk/work/conda/etc/profile.d/conda.sh
conda activate fastai-v2
python -m fastai.launch --gpus 0 /data/users/bressekk/work/faimed3d/pretrained/pretrain-ucf-101.py
