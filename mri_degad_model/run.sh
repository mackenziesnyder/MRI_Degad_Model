#!/bin/bash
#SBATCH --account=ctb-akhanf
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpu=l40s:1
#SBATCH --time=1-17:00:00

module load python/3.11.5
source ./venv/bin/activate
python3 new_CNN.py --input ../degad_preprocessed_data/ --image_size 255 --batch_size 42 --lr 0.0001 --filter 64 --depth 4 --loss mae --output_dir ./output

