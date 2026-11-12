#!/bin/bash
#SBATCH --partition=yoda
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# --per-class 200 to match sample.py's reduced sample count (see
# job_sample.sh). IMPORTANT: this means FID here is NOT directly comparable
# to hpc_experiment's 1000/class result on sample-size grounds alone (FID is
# known to be biased upward at smaller N) -- note this when comparing.
#
# pytorch-fid downloads InceptionV3 weights on first use; already pre-warmed
# in ~/.cache/torch/hub/checkpoints/ from the earlier experiments, so this
# should just work.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

MYDATASET="hest-bench"
DATASET_DST=/var/local/storage/${USER}/${MYDATASET}
if [ ! -d ${DATASET_DST} ]; then
    cp -a /opt/storage/users/${USER}/${MYDATASET} ${DATASET_DST}
fi

EXP_DIR=/var/local/storage/${USER}/experiments/hest-large
PERSIST_DIR=/opt/storage/users/${USER}/experiments/hest-large
mkdir -p ${EXP_DIR}/real

python3 ~/hest-large/compute_fid.py \
    --hest-dir ${DATASET_DST} \
    --synthetic-dir ${PERSIST_DIR}/synthetic \
    --real-dir ${EXP_DIR}/real \
    --per-class 200 \
    --results-json ${EXP_DIR}/fid_results.json \
    "$@"

cp ${EXP_DIR}/fid_results.json ${PERSIST_DIR}/fid_results.json
