#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --time=01:00:00

set -e  # propagate python/sbatch-step failures -- without this, a crashed
        # job falls through and SLURM reports false success

# No --gres line: pure network I/O. Only ~12.5GB (st/*.h5ad for 108
# candidate WSIs), not the full ~242GB of images -- downloaded first so
# we can compute extraction-aware patch counts before committing to
# which 10 WSIs/type to keep and before downloading any WSI images.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

OUT_DIR=/opt/storage/datasets/HEST
mkdir -p ${OUT_DIR}

python3 ~/git/MM-SDG/hest_dataset_prep/download_adata_only.py \
    --metadata-csv ${OUT_DIR}/HEST_v1_3_0.csv \
    --output-dir ${OUT_DIR}

echo "Current size:"
du -sh ${OUT_DIR}
