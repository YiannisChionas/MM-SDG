#!/bin/bash
#SBATCH --partition=yoda
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -e  # propagate python/sbatch-step failures -- without this, a crashed
        # job falls through to cp -a below and SLURM reports false success

# --per-class 100 to match sample_sr.py. Real patches exported at NATIVE
# 224x224 (no downsampling) -- this is the whole point of this experiment:
# FID here only needs a 224->299 Inception resize (1.33x), not
# hpc_experiment's 64->299 (4.67x).

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

MYDATASET="hest-bench"
DATASET_DST=/var/local/storage/${USER}/${MYDATASET}
if [ ! -d ${DATASET_DST} ]; then
    cp -a /opt/storage/users/${USER}/${MYDATASET} ${DATASET_DST}
fi

EXP_DIR=/var/local/storage/${USER}/experiments/hest-sr
PERSIST_DIR=/opt/storage/users/${USER}/experiments/hest-sr
mkdir -p ${EXP_DIR}/real

python3 ~/hest-sr/compute_fid_sr.py \
    --hest-dir ${DATASET_DST} \
    --synthetic-dir ${PERSIST_DIR}/synthetic \
    --real-dir ${EXP_DIR}/real \
    --per-class 100 \
    --results-json ${EXP_DIR}/fid_results.json \
    "$@"

cp ${EXP_DIR}/fid_results.json ${PERSIST_DIR}/fid_results.json
