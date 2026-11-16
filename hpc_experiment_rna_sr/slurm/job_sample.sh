#!/bin/bash
#SBATCH --partition=yoda
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=04:45:00

set -ex  # -x: print every command before running it -- temporary, for
         # debugging a mysterious instant 0-1s failure with zero output

# Cascaded paired sampling on held-out WSIs: stage-1 (hpc_experiment_rna
# checkpoint) generates 64x64 conditioned on real expression, upsampled +
# fed into stage-2 (this experiment's checkpoint) for 224x224. 2000 samples
# (matching hpc_experiment_rna's own --max-samples for a fair before/after
# comparison), batch=32 -- estimated ~4.1h for stage-2's forward passes
# alone (236.9ms/step locally), fits the 4h45m limit with little margin;
# resubmit the same command to resume if it times out.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

MYDATASET="hest-bench"
DATASET_DST=/var/local/storage/${USER}/${MYDATASET}
if [ ! -d ${DATASET_DST} ]; then
    cp -a /opt/storage/users/${USER}/${MYDATASET} ${DATASET_DST}
fi

PERSIST_DIR=/opt/storage/users/${USER}/experiments/hest-rna-sr
RNA_PERSIST_DIR=/opt/storage/users/${USER}/experiments/hest-rna
OUT=/var/local/storage/${USER}/experiments/hest-rna-sr/samples
PERSIST_OUT=${PERSIST_DIR}/samples

mkdir -p ${OUT}
mkdir -p ${PERSIST_OUT}

if [ -d ${PERSIST_OUT} ]; then
    cp -a ${PERSIST_OUT}/. ${OUT}/ 2>/dev/null || true
fi

python3 ~/hest-rna-sr/sample_rna_sr.py \
    --data-dir ${DATASET_DST} \
    --vae-checkpoint ${RNA_PERSIST_DIR}/vae_checkpoint.pt \
    --stage1-checkpoint ${RNA_PERSIST_DIR}/checkpoint.pt \
    --stage2-checkpoint ${PERSIST_DIR}/checkpoint.pt \
    --output-dir ${OUT} \
    --fold 0 \
    --max-samples 2000 \
    --batch-size 32 \
    "$@"

cp -a ${OUT}/. ${PERSIST_OUT}/
