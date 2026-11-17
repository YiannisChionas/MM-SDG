#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -e  # propagate python/sbatch-step failures -- without this, a crashed
        # job falls through to cp -a below and SLURM reports false success

# Small CNN regressor (4 conv layers) -- a MIG slice is plenty, please
# don't request yoda/chewbacca for this. Pass --image-size 64 or 224
# (must match the resolution of whatever synthetic samples it will judge)
# via "$@", e.g.:
#   sbatch job_train_regressor.sh --image-size 64 --epochs 15
#   sbatch job_train_regressor.sh --image-size 224 --epochs 15

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

MYDATASET="hest-bench"
DATASET_DST=/var/local/storage/${USER}/${MYDATASET}
if [ ! -d ${DATASET_DST} ]; then
    cp -a /opt/storage/users/${USER}/${MYDATASET} ${DATASET_DST}
fi

EXP_DIR=/var/local/storage/${USER}/experiments/hest-regressor
PERSIST_DIR=/opt/storage/users/${USER}/experiments/hest-regressor
mkdir -p ${EXP_DIR} ${PERSIST_DIR}

# Resume from persistent storage if a checkpoint already exists there
cp -a ${PERSIST_DIR}/. ${EXP_DIR}/ 2>/dev/null || true

python3 ~/hest-regressor/train_regressor.py \
    --data-dir ${DATASET_DST} \
    --output-dir ${EXP_DIR} \
    --fold 0 \
    --batch-size 64 \
    --lr 1e-4 \
    --num-workers 8 \
    "$@"

cp -a ${EXP_DIR}/. ${PERSIST_DIR}/
