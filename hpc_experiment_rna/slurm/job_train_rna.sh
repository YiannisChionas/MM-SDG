#!/bin/bash
#SBATCH --partition=yoda
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# CCRCC train-split only (20/24 WSIs) -- smaller than the pooled 10-class
# experiment, should comfortably fit several epochs in one job. Requires
# vae_checkpoint.pt from job_train_vae.sh to already exist.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

MYDATASET="hest-bench"
DATASET_SRC=/opt/storage/users/${USER}/${MYDATASET}
DATASET_DST=/var/local/storage/${USER}/${MYDATASET}
if [ ! -d ${DATASET_DST} ]; then
    cp -a ${DATASET_SRC} ${DATASET_DST}
fi

EXP_DIR=/var/local/storage/${USER}/experiments/hest-rna
PERSIST_DIR=/opt/storage/users/${USER}/experiments/hest-rna
mkdir -p ${EXP_DIR} ${PERSIST_DIR}

if [ -f ${PERSIST_DIR}/checkpoint.pt ] && [ ! -f ${EXP_DIR}/checkpoint.pt ]; then
    cp ${PERSIST_DIR}/checkpoint.pt ${EXP_DIR}/checkpoint.pt
fi
if [ ! -f ${EXP_DIR}/vae_checkpoint.pt ]; then
    cp ${PERSIST_DIR}/vae_checkpoint.pt ${EXP_DIR}/vae_checkpoint.pt
fi

python3 ~/hest-rna/train_rna.py \
    --data-dir ${DATASET_DST} \
    --vae-checkpoint ${EXP_DIR}/vae_checkpoint.pt \
    --output-dir ${EXP_DIR} \
    --fold 0 \
    --batch-size 64 \
    --lr 1e-4 \
    --num-workers 8 \
    "$@"

cp -a ${EXP_DIR}/. ${PERSIST_DIR}/
