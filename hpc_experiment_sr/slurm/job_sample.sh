#!/bin/bash
#SBATCH --partition=yoda
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

# Cascaded sampling: stage-1 (hpc_experiment checkpoint) generates 64x64,
# upsampled, fed into stage-2 (this experiment's checkpoint) for 224x224.
# 100/class (1000 total), batch=32 -- measured forward-only cost locally
# (236.9ms/step at batch=32) -> ~2.1h estimated, fits in 3h with margin.
# Resumable: rerun with the same --output-dir to continue after a timeout.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

STAGE1_CKPT=/opt/storage/users/${USER}/experiments/hest-cdm/checkpoint.pt
STAGE2_CKPT=/opt/storage/users/${USER}/experiments/hest-sr/checkpoint.pt
OUT=/var/local/storage/${USER}/experiments/hest-sr/synthetic
PERSIST_OUT=/opt/storage/users/${USER}/experiments/hest-sr/synthetic

mkdir -p ${OUT}
mkdir -p ${PERSIST_OUT}

if [ -d ${PERSIST_OUT} ]; then
    cp -a ${PERSIST_OUT}/. ${OUT}/ 2>/dev/null || true
fi

python3 ~/hest-sr/sample_sr.py \
    --stage1-checkpoint ${STAGE1_CKPT} \
    --stage2-checkpoint ${STAGE2_CKPT} \
    --output-dir ${OUT} \
    --per-class 100 \
    --batch-size 32 \
    "$@"

cp -a ${OUT}/. ${PERSIST_OUT}/
