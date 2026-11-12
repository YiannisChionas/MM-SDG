#!/bin/bash
#SBATCH --partition=yoda
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=03:30:00

# Reduced to 200/class (2000 total, vs 1000/class for the small model) --
# the large model's per-step cost makes 10,000 images infeasible within the
# 5h job limit (estimated >14h at full scale). 2000 should fit in ~2.5h at
# batch=64, leaving margin. This means the large-vs-small FID comparison
# won't be apples-to-apples on sample count -- FID is known to be biased at
# smaller sample sizes, so compare with that caveat in mind (see README).
# Resumable: rerun with the same --output-dir to continue after a timeout.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

CKPT=/opt/storage/users/${USER}/experiments/hest-large/checkpoint.pt
OUT=/var/local/storage/${USER}/experiments/hest-large/synthetic
PERSIST_OUT=/opt/storage/users/${USER}/experiments/hest-large/synthetic

mkdir -p ${OUT}
mkdir -p ${PERSIST_OUT}

if [ -d ${PERSIST_OUT} ]; then
    cp -a ${PERSIST_OUT}/. ${OUT}/ 2>/dev/null || true
fi

python3 ~/hest-large/sample.py \
    --checkpoint ${CKPT} \
    --output-dir ${OUT} \
    --per-class 200 \
    --batch-size 64 \
    "$@"

cp -a ${OUT}/. ${PERSIST_OUT}/
