#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --time=00:10:00

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

python3 ~/git/MM-SDG/hest_dataset_prep/survey_alternatives.py \
    --metadata-csv /opt/storage/datasets/HEST/HEST_v1_3_0.csv \
    --min-samples 5
