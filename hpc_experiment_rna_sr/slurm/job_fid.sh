#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -e

# Lightweight (Inception features only) -- obiwan's MIG is fine here, the
# compute-restriction problem only bit us on actual diffusion training/
# sampling, not a single feature-extraction pass over ~4000 images.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

PERSIST_DIR=/opt/storage/users/${USER}/experiments/hest-rna-sr

python3 ~/hest-rna-sr/compute_fid_rna_sr.py \
    --samples-dir ${PERSIST_DIR}/samples \
    --results-json ${PERSIST_DIR}/fid_rna_sr_results.json \
    "$@"
