#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

set -e  # propagate python/sbatch-step failures -- without this, a crashed
        # job falls through to cp -a below and SLURM reports false success

# CPU-only -- no GPU line, this is a scanpy preprocessing pass (standard
# HVG selection) over the full ~36,601-gene transcriptome for the CCRCC
# train-split WSIs, not a model. Run once; the resulting gene panel
# (whatever size it comes out to -- NOT forced to 17,655, see README) is
# then fixed for train_vae_paper.py.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg
pip install -q scanpy  # not part of the base env yet

MYDATASET="hest-bench"
DATASET_DST=/var/local/storage/${USER}/${MYDATASET}
if [ ! -d ${DATASET_DST} ]; then
    cp -a /opt/storage/users/${USER}/${MYDATASET} ${DATASET_DST}
fi

EXP_DIR=/var/local/storage/${USER}/experiments/hest-vae-paper
PERSIST_DIR=/opt/storage/users/${USER}/experiments/hest-vae-paper
mkdir -p ${EXP_DIR} ${PERSIST_DIR}

python3 ~/hest-vae-paper/select_genes.py \
    --data-dir ${DATASET_DST} \
    --output-dir ${EXP_DIR} \
    --fold 0 \
    "$@"

cp -a ${EXP_DIR}/. ${PERSIST_DIR}/
