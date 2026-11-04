#!/bin/bash
#SBATCH --partition=yoda
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

# Paired generation on held-out (test-split) WSIs, capped at 2000 spots by
# default -- resubmit the same command to resume if it times out.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

MYDATASET="hest-bench"
DATASET_DST=/var/local/storage/${USER}/${MYDATASET}
if [ ! -d ${DATASET_DST} ]; then
    cp -a /opt/storage/users/${USER}/${MYDATASET} ${DATASET_DST}
fi

PERSIST_DIR=/opt/storage/users/${USER}/experiments/hest-rna
EXP_DIR=/var/local/storage/${USER}/experiments/hest-rna
OUT=${EXP_DIR}/samples
PERSIST_OUT=${PERSIST_DIR}/samples

mkdir -p ${OUT} ${PERSIST_OUT}

if [ -d ${PERSIST_OUT} ]; then
    cp -a ${PERSIST_OUT}/. ${OUT}/ 2>/dev/null || true
fi

python3 ~/hest-rna/sample_rna.py \
    --data-dir ${DATASET_DST} \
    --vae-checkpoint ${PERSIST_DIR}/vae_checkpoint.pt \
    --diffusion-checkpoint ${PERSIST_DIR}/checkpoint.pt \
    --output-dir ${OUT} \
    --fold 0 \
    --max-samples 2000 \
    --batch-size 128 \
    "$@"

cp -a ${OUT}/. ${PERSIST_OUT}/
