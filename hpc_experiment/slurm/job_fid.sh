#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Lightweight: just Inception feature extraction over 20K images (10K real +
# 10K synthetic). Uses obiwan (MIG A100) since full A100/A6000 muscle isn't
# needed here -- please don't request yoda/chewbacca for this.
#
# IMPORTANT: pytorch-fid downloads InceptionV3 weights on first use. If this
# job fails with a download/connection error, pre-warm the weight cache from
# an interactive CPU session first (see compute_fid.py docstring).

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

MYDATASET="hest-bench"
DATASET_DST=/var/local/storage/${USER}/${MYDATASET}
if [ ! -d ${DATASET_DST} ]; then
    cp -a /opt/storage/users/${USER}/${MYDATASET} ${DATASET_DST}
fi

EXP_DIR=/var/local/storage/${USER}/experiments/hest-cdm
PERSIST_DIR=/opt/storage/users/${USER}/experiments/hest-cdm
mkdir -p ${EXP_DIR}/real

python3 ~/hest-cdm/compute_fid.py \
    --hest-dir ${DATASET_DST} \
    --synthetic-dir ${PERSIST_DIR}/synthetic \
    --real-dir ${EXP_DIR}/real \
    --per-class 1000 \
    --results-json ${EXP_DIR}/fid_results.json \
    "$@"

cp ${EXP_DIR}/fid_results.json ${PERSIST_DIR}/fid_results.json
