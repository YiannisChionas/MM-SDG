#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -e  # propagate python/sbatch-step failures -- without this, a crashed
        # job falls through to cp -a below and SLURM reports false success

# Small CNN classifier (4 conv layers) -- a MIG slice is plenty, please
# don't request yoda/chewbacca for this. Pass --image-size 64 or 224
# (must match the resolution of whatever synthetic samples it will judge)
# via "$@", e.g.:
#   sbatch job_train_classifier.sh --image-size 64 --epochs 15
#   sbatch job_train_classifier.sh --image-size 224 --epochs 15

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

MYDATASET="hest-bench"
DATASET_DST=/var/local/storage/${USER}/${MYDATASET}
if [ ! -d ${DATASET_DST} ]; then
    cp -a /opt/storage/users/${USER}/${MYDATASET} ${DATASET_DST}
fi

EXP_DIR=/var/local/storage/${USER}/experiments/hest-classifier
PERSIST_DIR=/opt/storage/users/${USER}/experiments/hest-classifier
mkdir -p ${EXP_DIR} ${PERSIST_DIR}

# Resume from persistent storage if a checkpoint already exists there
cp -a ${PERSIST_DIR}/. ${EXP_DIR}/ 2>/dev/null || true

python3 ~/hest-classifier/train_classifier.py \
    --data-dir ${DATASET_DST} \
    --output-dir ${EXP_DIR} \
    --batch-size 64 \
    --lr 1e-4 \
    --num-workers 8 \
    "$@"

cp -a ${EXP_DIR}/. ${PERSIST_DIR}/
