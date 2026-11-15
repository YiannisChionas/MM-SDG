#!/bin/bash
#SBATCH --partition=yoda
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:45:00

set -e  # propagate python failures -- without this, a crashed training
        # run still falls through to `cp -a` below and SLURM reports
        # false success (this bit us once already: OOM crashed silently).

# Switched back to yoda (full A100 40GB) -- obiwan's MIG slices turned out
# ~2.8x slower on compute, not just memory-limited (244.2 min/epoch
# measured vs ~87min/epoch estimated). batch=32 here, matching the
# original design and hpc_experiment's demonstrated real throughput on
# this partition (which beat its own local-benchmark estimate).
#
# --persist-dir still copies checkpoint.pt back to persistent storage
# after EVERY epoch (not just at clean job exit) -- keep this regardless
# of partition, it's cheap insurance against a TIMEOUT kill.

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
    --persist-dir ${PERSIST_DIR} \
    --batch-size 32 \
    --lr 1e-4 \
    --num-workers 8 \
    "$@"

cp -a ${EXP_DIR}/. ${PERSIST_DIR}/
