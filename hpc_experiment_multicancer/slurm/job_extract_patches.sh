#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:45:00

set -e  # propagate python/sbatch-step failures -- without this, a crashed
        # job falls through and SLURM reports false success

# CPU-only -- cropping/resizing, no GPU needed. ~30-40ms/crop measured,
# up to 4000 crops/WSI -> up to ~160s/WSI just for cropping; 50 WSIs
# could approach the 5h limit in one job. Resumable (skips a WSI if its
# output .h5 already exists) -- pass --oncotree-code to split across
# multiple parallel jobs instead, e.g.:
#   sbatch job_extract_patches.sh --oncotree-code COAD
#   sbatch job_extract_patches.sh --oncotree-code SCCRCC
#   ... (one job per type, all fit well within the 2-job concurrent cap
#   two at a time)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg
pip install -q tifffile imagecodecs zarr h5py  # not part of the base env yet

HEST_DIR=/opt/storage/datasets/HEST
OUT_DIR=${HEST_DIR}/patches_multicancer

python3 ~/git/MM-SDG/hpc_experiment_multicancer/extract_patches.py \
    --selected-wsis-csv ${HEST_DIR}/selected_wsis.csv \
    --hest-dir ${HEST_DIR} \
    --output-dir ${OUT_DIR} \
    "$@"

echo "Current size:"
du -sh ${OUT_DIR}
