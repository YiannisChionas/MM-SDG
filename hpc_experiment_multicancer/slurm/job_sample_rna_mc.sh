#!/bin/bash
#SBATCH --partition=anakin
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -e  # propagate python/sbatch-step failures -- without this, a crashed
        # job falls through to cp -a below and SLURM reports false success

# Paired generation on held-out (test-split) WSIs, using the EMA weights
# and the Karras/Heun sampler (~40 steps instead of 1000) -- see
# sample_rna_mc.py docstring. anakin access is temporary, requires explicit
# go-ahead each time, per CLAUDE.md Rule #1.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

HEST_DIR=/opt/storage/datasets/HEST
EXP_DIR=/opt/storage/users/${USER}/experiments/hest-rna-multicancer
OUT=${EXP_DIR}/samples
mkdir -p ${OUT}

python3 ~/git/MM-SDG/hpc_experiment_multicancer/sample_rna_mc.py \
    --hest-dir ${HEST_DIR} \
    --selected-wsis-csv ${HEST_DIR}/selected_wsis.csv \
    --vae-checkpoint /opt/storage/users/${USER}/experiments/hest-vae-multicancer/vae_multicancer_best.pt \
    --diffusion-checkpoint ${EXP_DIR}/checkpoint.pt \
    --output-dir ${OUT} \
    --max-samples 2000 \
    --batch-size 64 \
    --num-inference-steps 40 \
    "$@"
