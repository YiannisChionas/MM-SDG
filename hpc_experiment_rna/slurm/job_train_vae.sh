#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

# Tiny model (50-dim MLP VAE) -- a small MIG slice is plenty, please don't
# request yoda/chewbacca for this.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

MYDATASET="hest-bench"
DATASET_DST=/var/local/storage/${USER}/${MYDATASET}
if [ ! -d ${DATASET_DST} ]; then
    cp -a /opt/storage/users/${USER}/${MYDATASET} ${DATASET_DST}
fi

EXP_DIR=/var/local/storage/${USER}/experiments/hest-rna
PERSIST_DIR=/opt/storage/users/${USER}/experiments/hest-rna
mkdir -p ${EXP_DIR} ${PERSIST_DIR}

python3 ~/hest-rna/train_vae.py \
    --data-dir ${DATASET_DST} \
    --output-dir ${EXP_DIR} \
    --fold 0 \
    --epochs 50 \
    "$@"

cp -a ${EXP_DIR}/. ${PERSIST_DIR}/
