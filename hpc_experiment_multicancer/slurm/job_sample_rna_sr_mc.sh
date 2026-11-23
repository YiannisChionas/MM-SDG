#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -e  # propagate python/sbatch-step failures -- without this, a crashed
        # job falls through to cp -a below and SLURM reports false success

# Full-cascade paired generation (stage 1 -> blur -> stage 2) on held-out
# (test-split) WSIs -- see sample_rna_sr_mc.py docstring. Heavier than
# stage-1-only sampling (two models, 256x256 output) but still much
# lighter than training, so obiwan rather than anakin.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

HEST_DIR=/opt/storage/datasets/HEST
STAGE1_DIR=/opt/storage/users/${USER}/experiments/hest-rna-multicancer
STAGE2_DIR=/opt/storage/users/${USER}/experiments/hest-rna-sr-multicancer
OUT=${STAGE2_DIR}/samples
mkdir -p ${OUT}

python3 ~/git/MM-SDG/hpc_experiment_multicancer/sample_rna_sr_mc.py \
    --hest-dir ${HEST_DIR} \
    --selected-wsis-csv ${HEST_DIR}/selected_wsis.csv \
    --vae-checkpoint /opt/storage/users/${USER}/experiments/hest-vae-multicancer/vae_multicancer_best.pt \
    --stage1-checkpoint ${STAGE1_DIR}/checkpoint.pt \
    --stage2-checkpoint ${STAGE2_DIR}/checkpoint.pt \
    --output-dir ${OUT} \
    --max-samples 2000 \
    --batch-size 32 \
    "$@"
