#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

OUT_DIR=/opt/storage/datasets/HEST

python3 ~/git/MM-SDG/hest_dataset_prep/compute_extractable_patches.py \
    --metadata-csv ${OUT_DIR}/HEST_v1_3_0.csv \
    --adata-dir ${OUT_DIR} \
    --patch-size-um 128 \
    --output-csv ${OUT_DIR}/extractable_patches.csv
