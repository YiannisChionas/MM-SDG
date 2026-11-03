#!/bin/bash
#SBATCH --partition=yoda
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# ~3.2h estimated for 10 epochs over the pooled 236K-patch dataset (fp32,
# batch=64) -- 4h leaves margin under the cluster's 5h hard limit. For more
# epochs, use submit_train.sh instead of calling this directly.

# Activate Anaconda work environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

# Stage dataset to fast local scratch (per cluster storage guidelines)
MYDATASET="hest-bench"
DATASET_SRC=/opt/storage/users/${USER}/${MYDATASET}
DATASET_DST=/var/local/storage/${USER}/${MYDATASET}
if [ ! -d ${DATASET_DST} ]; then
    cp -a ${DATASET_SRC} ${DATASET_DST}
fi

EXP_DIR=/var/local/storage/${USER}/experiments/hest-cdm
PERSIST_DIR=/opt/storage/users/${USER}/experiments/hest-cdm
mkdir -p ${EXP_DIR}
mkdir -p ${PERSIST_DIR}

# Resume automatically: train.py reuses ${EXP_DIR}/checkpoint.pt if present.
# Copy down any existing checkpoint from persistent storage first so repeated
# submissions (e.g. via submit_train.sh) continue training instead of
# restarting from scratch.
if [ -f ${PERSIST_DIR}/checkpoint.pt ] && [ ! -f ${EXP_DIR}/checkpoint.pt ]; then
    cp ${PERSIST_DIR}/checkpoint.pt ${EXP_DIR}/checkpoint.pt
fi

python3 ~/hest-cdm/train.py \
    --data-dir ${DATASET_DST} \
    --output-dir ${EXP_DIR} \
    --batch-size 64 \
    --lr 1e-4 \
    --num-workers 8 \
    "$@"

# Copy results back to persistent storage -- /var/local/storage is per-node
# scratch and not guaranteed to survive between jobs/nodes.
cp -a ${EXP_DIR}/. ${PERSIST_DIR}/
