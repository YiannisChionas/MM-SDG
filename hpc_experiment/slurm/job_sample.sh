#!/bin/bash
#SBATCH --partition=yoda
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

# ~2.1h estimated for 10,000 images (1000/class x 10 classes) at batch=128,
# fp32, 1000-step reverse diffusion each. 3h leaves margin.
# Resumable: rerun with the same --output-dir to continue after a timeout.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

CKPT=/opt/storage/users/${USER}/experiments/hest-cdm/checkpoint.pt
OUT=/var/local/storage/${USER}/experiments/hest-cdm/synthetic
PERSIST_OUT=/opt/storage/users/${USER}/experiments/hest-cdm/synthetic

mkdir -p ${OUT}
mkdir -p ${PERSIST_OUT}

# Pick up any partial run already copied back from a previous (timed-out) job
if [ -d ${PERSIST_OUT} ]; then
    cp -a ${PERSIST_OUT}/. ${OUT}/ 2>/dev/null || true
fi

python3 ~/hest-cdm/sample.py \
    --checkpoint ${CKPT} \
    --output-dir ${OUT} \
    --per-class 1000 \
    --batch-size 128 \
    "$@"

cp -a ${OUT}/. ${PERSIST_OUT}/
