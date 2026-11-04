#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Lightweight: Inception features over ~4000 images (2000 real + 2000 synth).
# Pre-warm weights first if this fails on a download/connection error --
# see compute_fid_rna.py docstring.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

PERSIST_DIR=/opt/storage/users/${USER}/experiments/hest-rna

python3 ~/hest-rna/compute_fid_rna.py \
    --samples-dir ${PERSIST_DIR}/samples \
    --results-json ${PERSIST_DIR}/fid_rna_results.json \
    "$@"
