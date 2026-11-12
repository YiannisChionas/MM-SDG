#!/bin/bash
#SBATCH --partition=yoda
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# LARGE model (454.5M params, ~4x hpc_experiment's 114M). Measured locally:
# batch=32 -> 535ms/step, 14.75GB peak VRAM on a 16GB card. Extrapolated to
# batch=64 (untestable locally -- would OOM): ~1070ms/step, ~3.6x slower
# than the small model's measured 297ms/step at the same batch size. That
# first local batch=64 test actually measured 18.6s/step (62x) -- that was
# VRAM-exhaustion thrashing on the 16GB local card, not real compute cost;
# a 40GB A100 should have enough headroom to avoid it. Still, this estimate
# is unverified at cluster scale -- watch the first epoch's timing.
#
# At ~40min/epoch (11min x 3.6), 10 epochs would need ~6.7h -- over the 5h
# limit. Use submit_train.sh (5 epochs/job x 2 jobs) instead of calling this
# directly for a full 10-epoch run.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

MYDATASET="hest-bench"
DATASET_SRC=/opt/storage/users/${USER}/${MYDATASET}
DATASET_DST=/var/local/storage/${USER}/${MYDATASET}
if [ ! -d ${DATASET_DST} ]; then
    cp -a ${DATASET_SRC} ${DATASET_DST}
fi

EXP_DIR=/var/local/storage/${USER}/experiments/hest-large
PERSIST_DIR=/opt/storage/users/${USER}/experiments/hest-large
mkdir -p ${EXP_DIR}
mkdir -p ${PERSIST_DIR}

if [ -f ${PERSIST_DIR}/checkpoint.pt ] && [ ! -f ${EXP_DIR}/checkpoint.pt ]; then
    cp ${PERSIST_DIR}/checkpoint.pt ${EXP_DIR}/checkpoint.pt
fi

python3 ~/hest-large/train.py \
    --data-dir ${DATASET_DST} \
    --output-dir ${EXP_DIR} \
    --batch-size 64 \
    --lr 1e-4 \
    --num-workers 8 \
    "$@"

cp -a ${EXP_DIR}/. ${PERSIST_DIR}/
