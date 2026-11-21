#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

set -e  # propagate python/sbatch-step failures -- without this, a crashed
        # job falls through and SLURM reports false success

# CPU-only -- scanpy QC filtering over pooled spots, no GPU/model.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg
pip install -q scanpy  # not part of the base env yet

HEST_DIR=/opt/storage/datasets/HEST

python3 ~/git/MM-SDG/hpc_experiment_multicancer/select_genes_multicancer.py \
    --selected-wsis-csv ${HEST_DIR}/selected_wsis.csv \
    --adata-dir ${HEST_DIR} \
    --output-dir ${HEST_DIR}
