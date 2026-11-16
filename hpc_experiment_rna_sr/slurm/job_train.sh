#!/bin/bash
#SBATCH --partition=yoda
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -e  # propagate python failures -- see hpc_experiment_sr's job_train.sh
        # for why this matters (a crashed run would otherwise fall through
        # to cp -a below and SLURM would report false success).

# CCRCC train-split only (~55,808 patches, measured from hpc_experiment_rna's
# own run) -- much smaller than hpc_experiment_sr's pooled 236,495. Scaling
# that experiment's measured yoda throughput (51min/epoch for 236,495
# patches): ~12min/epoch estimated here. 10 epochs -> ~2h, comfortable
# single-job margin -- unlike the other SR experiment, this one likely
# doesn't need submit_train.sh-style chaining, but --persist-dir is kept
# anyway as cheap insurance against a TIMEOUT kill.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

MYDATASET="hest-bench"
DATASET_SRC=/opt/storage/users/${USER}/${MYDATASET}
DATASET_DST=/var/local/storage/${USER}/${MYDATASET}
if [ ! -d ${DATASET_DST} ]; then
    cp -a ${DATASET_SRC} ${DATASET_DST}
fi

EXP_DIR=/var/local/storage/${USER}/experiments/hest-rna-sr
PERSIST_DIR=/opt/storage/users/${USER}/experiments/hest-rna-sr
mkdir -p ${EXP_DIR}
mkdir -p ${PERSIST_DIR}

if [ -f ${PERSIST_DIR}/checkpoint.pt ] && [ ! -f ${EXP_DIR}/checkpoint.pt ]; then
    cp ${PERSIST_DIR}/checkpoint.pt ${EXP_DIR}/checkpoint.pt
fi
if [ ! -f ${EXP_DIR}/vae_checkpoint.pt ]; then
    cp /opt/storage/users/${USER}/experiments/hest-rna/vae_checkpoint.pt ${EXP_DIR}/vae_checkpoint.pt
fi

accelerate launch --num_processes=1 ~/hest-rna-sr/train_rna_sr.py \
    --data-dir ${DATASET_DST} \
    --vae-checkpoint ${EXP_DIR}/vae_checkpoint.pt \
    --output-dir ${EXP_DIR} \
    --persist-dir ${PERSIST_DIR} \
    --fold 0 \
    --batch-size 32 \
    --lr 1e-4 \
    --num-workers 8 \
    "$@"

cp -a ${EXP_DIR}/. ${PERSIST_DIR}/
