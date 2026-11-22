#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -e  # propagate python/sbatch-step failures -- without this, a crashed
        # job falls through to cp -a below and SLURM reports false success

# Lightweight: Inception features over a real/synthetic pair produced by
# sample_rna_mc.py. Pre-warm weights first if this fails on a download/
# connection error -- see compute_fid_mc.py docstring.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

EXP_DIR=/opt/storage/users/${USER}/experiments/hest-rna-multicancer

python3 ~/git/MM-SDG/hpc_experiment_multicancer/compute_fid_mc.py \
    --results-json ${EXP_DIR}/fid_mc_results.json \
    "$@"
