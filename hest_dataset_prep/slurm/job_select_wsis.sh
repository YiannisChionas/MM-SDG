#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --time=00:10:00

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

python3 ~/git/MM-SDG/hest_dataset_prep/select_wsis.py \
    --metadata-csv /opt/storage/datasets/HEST/HEST_v1_3_0.csv \
    --n-per-type 10 \
    --output-csv /opt/storage/datasets/HEST/selected_wsis.csv
