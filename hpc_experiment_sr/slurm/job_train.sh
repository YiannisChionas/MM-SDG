#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:45:00

set -e  # propagate python failures -- without this, a crashed training
        # run still falls through to `cp -a` below and SLURM reports
        # false success (this bit us once already: OOM crashed silently).

# Single MIG A100 20GB. The 2-GPU accelerate attempt hit a cluster-side
# MIG/NCCL issue (both ranks resolved to the SAME physical device --
# "Duplicate GPU detected" -- a SLURM/MIG gres config problem, not fixable
# from this script) -- reverted to single-GPU until that's sorted out.
# batch=8 measured locally at 7.19GB, comfortable margin under 20GB.
# train_sr.py's Accelerator() works transparently in single-process mode,
# no script changes needed, just --num_processes=1.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

MYDATASET="hest-bench"
DATASET_SRC=/opt/storage/users/${USER}/${MYDATASET}
DATASET_DST=/var/local/storage/${USER}/${MYDATASET}
if [ ! -d ${DATASET_DST} ]; then
    cp -a ${DATASET_SRC} ${DATASET_DST}
fi

EXP_DIR=/var/local/storage/${USER}/experiments/hest-sr
PERSIST_DIR=/opt/storage/users/${USER}/experiments/hest-sr
mkdir -p ${EXP_DIR}
mkdir -p ${PERSIST_DIR}

if [ -f ${PERSIST_DIR}/checkpoint.pt ] && [ ! -f ${EXP_DIR}/checkpoint.pt ]; then
    cp ${PERSIST_DIR}/checkpoint.pt ${EXP_DIR}/checkpoint.pt
fi

accelerate launch --num_processes=1 ~/hest-sr/train_sr.py \
    --data-dir ${DATASET_DST} \
    --output-dir ${EXP_DIR} \
    --batch-size 8 \
    --lr 1e-4 \
    --num-workers 8 \
    "$@"

cp -a ${EXP_DIR}/. ${PERSIST_DIR}/
