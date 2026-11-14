#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=04:45:00

set -e  # propagate python failures -- without this, a crashed training
        # run still falls through to `cp -a` below and SLURM reports
        # false success (this bit us once already: OOM crashed silently).

# 2x MIG A100 20GB via accelerate (DDP), NOT 1x40GB -- obiwan's MIG slices
# don't combine into a bigger memory pool, so --batch-size here is PER GPU
# (effective batch = this x 2). batch=16/GPU measured locally at 14.0GB on
# a single GPU -- should fit a 20GB MIG with margin; watch the first epoch
# for OOM anyway since this is the first real multi-GPU run (untestable
# locally, only 1 GPU on the dev machine).
#
# Still ~87min/epoch single-GPU-equivalent-adjusted estimate -> 2 GPUs
# should roughly halve that, but unverified. Keep submit_train.sh's 3
# epochs/job chaining regardless, in case the speedup is less than hoped.

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

accelerate launch --multi_gpu --num_processes=2 ~/hest-sr/train_sr.py \
    --data-dir ${DATASET_DST} \
    --output-dir ${EXP_DIR} \
    --batch-size 16 \
    --lr 1e-4 \
    --num-workers 4 \
    "$@"

cp -a ${EXP_DIR}/. ${PERSIST_DIR}/
