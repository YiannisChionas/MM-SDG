#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:55:00

set -e  # propagate python failures -- without this, a crashed training
        # run still falls through to `cp -a` below and SLURM reports
        # false success (this bit us once already: OOM crashed silently).

# Single MIG A100 20GB. Measured for real (not estimated): epoch 0 took
# 244.2 min here -- ~2.8x slower than the ~87min/epoch estimated from a
# full RTX 5070 Ti locally. MIG slices apparently restrict compute (SMs),
# not just memory. At this rate only ~1 epoch fits per job even with the
# time limit raised close to the cluster's 5h max -- see submit_train.sh,
# now 1 epoch/job x 12 jobs instead of 3x4.
#
# --persist-dir makes train_sr.py copy checkpoint.pt back to persistent
# storage after EVERY epoch, not just at clean job exit -- a hard SLURM
# TIMEOUT kill (which happened once already) skips the `cp -a` below
# entirely, silently losing an already-completed epoch's progress.

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
    --batch-size 8 \
    --lr 1e-4 \
    --num-workers 8 \
    "$@"

cp -a ${EXP_DIR}/. ${PERSIST_DIR}/
