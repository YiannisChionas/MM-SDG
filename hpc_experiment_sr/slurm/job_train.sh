#!/bin/bash
#SBATCH --partition=yoda
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:45:00

# 224x224 is ~12x more pixels than hpc_experiment's 64x64 -- measured
# locally (28.5M-param SR UNet, batch=8): 177ms/step, ~87min/epoch on the
# full pooled dataset (236,495 patches). A 10-12 epoch run needs ~14-17h,
# way over the 5h limit -- MUST use submit_train.sh (3 epochs/job x 4 jobs),
# never call this directly for a full run.

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

python3 ~/hest-sr/train_sr.py \
    --data-dir ${DATASET_DST} \
    --output-dir ${EXP_DIR} \
    --batch-size 32 \
    --lr 1e-4 \
    --num-workers 8 \
    "$@"

cp -a ${EXP_DIR}/. ${PERSIST_DIR}/
