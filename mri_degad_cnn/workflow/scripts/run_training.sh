#!/bin/bash
#SBATCH --account=def-akhanf-ab
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=7-00:00
echo "begin"
source $(kpy _kpy_wrapper)
kpy load venv_train_degad
echo "beginning training"
cd /home/fogunsan/projects/ctb-akhanf/cfmm-bids/Lau/degad/snakemake/snakemake_CNN/
snakemake --cores 1
