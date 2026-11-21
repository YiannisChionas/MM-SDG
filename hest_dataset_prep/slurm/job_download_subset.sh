#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --time=04:45:00

set -e  # propagate python/sbatch-step failures -- without this, a crashed
        # job falls through and SLURM reports false success

# No --gres line: pure network I/O, no GPU/compute needed. Downloads
# straight to persistent/shared storage (/opt/storage/datasets), not
# scratch -- this only needs to happen once. Resumable: snapshot_download
# skips files already on disk, so just resubmit if this times out.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

OUT_DIR=/opt/storage/datasets/HEST
mkdir -p ${OUT_DIR}

python3 ~/git/MM-SDG/hest_dataset_prep/download_subset.py \
    --selected-wsis-csv ${OUT_DIR}/selected_wsis.csv \
    --output-dir ${OUT_DIR}

echo "Current size:"
du -sh ${OUT_DIR}
