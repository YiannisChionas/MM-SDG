#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --time=00:30:00

set -e  # propagate python/sbatch-step failures -- without this, a crashed
        # job falls through and SLURM reports false success

# No --gres line: this just reads a small metadata CSV over the network,
# no GPU/compute needed. Requires HF access already granted on the
# MahmoodLab/hest dataset page (gated dataset) and a cached token (same
# one used for the hest-bench download).

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg
pip install -q fsspec huggingface_hub  # ensure hf:// read support

OUT_DIR=/opt/storage/datasets/HEST
mkdir -p ${OUT_DIR}

python3 ~/git/MM-SDG/hest_dataset_prep/fetch_metadata.py \
    --output-csv ${OUT_DIR}/HEST_v1_3_0.csv
