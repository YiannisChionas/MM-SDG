#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -e  # propagate python/sbatch-step failures -- without this, a crashed
        # job falls through to cp -a below and SLURM reports false success

# Pass --classifier-checkpoint, --samples-dir, --output-json via "$@", e.g.
# for the 64x64 class-conditioned experiment:
#   sbatch job_eval_classifier.sh \
#       --classifier-checkpoint /opt/storage/users/$USER/experiments/hest-classifier/classifier_64.pt \
#       --samples-dir /opt/storage/users/$USER/experiments/hest-cdm/synthetic \
#       --output-json /opt/storage/users/$USER/experiments/hest-classifier/eval_cdm_64.json
# or for the 224x224 cascade:
#   sbatch job_eval_classifier.sh \
#       --classifier-checkpoint /opt/storage/users/$USER/experiments/hest-classifier/classifier_224.pt \
#       --samples-dir /opt/storage/users/$USER/experiments/hest-sr/synthetic \
#       --output-json /opt/storage/users/$USER/experiments/hest-classifier/eval_sr_224.json

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

MYDATASET="hest-bench"
DATASET_DST=/var/local/storage/${USER}/${MYDATASET}
if [ ! -d ${DATASET_DST} ]; then
    cp -a /opt/storage/users/${USER}/${MYDATASET} ${DATASET_DST}
fi

python3 ~/hest-classifier/eval_classifier.py \
    --data-dir ${DATASET_DST} \
    --batch-size 64 \
    "$@"
