#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -e  # propagate python/sbatch-step failures -- without this, a crashed
        # job falls through to cp -a below and SLURM reports false success

# Pass --regressor-checkpoint, --samples-dir, --output-json, --max-samples
# via "$@", e.g. for the 64x64 RNA-conditioned experiment:
#   sbatch job_eval_regressor.sh \
#       --regressor-checkpoint /opt/storage/users/$USER/experiments/hest-regressor/regressor_64.pt \
#       --samples-dir /opt/storage/users/$USER/experiments/hest-rna/samples \
#       --output-json /opt/storage/users/$USER/experiments/hest-regressor/eval_rna_64.json \
#       --max-samples 2000
# or for the 224x224 cascade:
#   sbatch job_eval_regressor.sh \
#       --regressor-checkpoint /opt/storage/users/$USER/experiments/hest-regressor/regressor_224.pt \
#       --samples-dir /opt/storage/users/$USER/experiments/hest-rna-sr/samples \
#       --output-json /opt/storage/users/$USER/experiments/hest-regressor/eval_rna_sr_224.json \
#       --max-samples 2000

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

MYDATASET="hest-bench"
DATASET_DST=/var/local/storage/${USER}/${MYDATASET}
if [ ! -d ${DATASET_DST} ]; then
    cp -a /opt/storage/users/${USER}/${MYDATASET} ${DATASET_DST}
fi

python3 ~/hest-regressor/eval_regressor.py \
    --data-dir ${DATASET_DST} \
    --fold 0 \
    --batch-size 64 \
    "$@"
