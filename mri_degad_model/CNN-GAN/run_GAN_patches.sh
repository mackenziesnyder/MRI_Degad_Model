#!/bin/bash
#SBATCH --account=ctb-akhanf
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gpu=v100:1
#SBATCH --time=3-00:00:00

module load python/3.11.5
source ./venv/bin/activate
python3 GAN_patches.py --input ../degad_preprocessed_data/ --image_size 256 --patch_size 64 --batch_size 16 --lr 0.0001 --filterG 32 --filterD 32 --depthG 4 --trainD 3 --loss mae --output_dir ./output

