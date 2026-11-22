#!/bin/bash
#SBATCH --partition=yoda
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00

set -e  # propagate python/sbatch-step failures -- without this, a crashed
        # job falls through to cp -a below and SLURM reports false success

# ~282M-param beta-VAE (6000/4000 hidden, 200 latent) -- NOT a tiny model
# like hpc_experiment_rna's VAE, do not move this to obiwan. Depending on
# final --batch-size this may need anakin/luke instead of yoda once that
# access request goes through (see repo root README) -- partition here
# is a placeholder until settings are finalized (see this experiment's
# README "Settings to decide together" section). Requires gene_panel_*.json
# from job_select_genes.sh to already exist.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

MYDATASET="hest-bench"
DATASET_DST=/var/local/storage/${USER}/${MYDATASET}
if [ ! -d ${DATASET_DST} ]; then
    cp -a /opt/storage/users/${USER}/${MYDATASET} ${DATASET_DST}
fi

EXP_DIR=/var/local/storage/${USER}/experiments/hest-vae-paper
PERSIST_DIR=/opt/storage/users/${USER}/experiments/hest-vae-paper
mkdir -p ${EXP_DIR} ${PERSIST_DIR}

cp -a ${PERSIST_DIR}/. ${EXP_DIR}/ 2>/dev/null || true

# Sort numerically by the gene-count suffix and take the LARGEST --
# plain `ls | head -1` is alphabetical, so a stale small-count panel from
# an earlier broken run would always win over the real, larger one.
GENE_PANEL=$(ls ${EXP_DIR}/gene_panel_*.json | sort -t_ -k3 -n -r | head -1)
if [ -z "${GENE_PANEL}" ]; then
    echo "No gene_panel_*.json found in ${EXP_DIR} -- run job_select_genes.sh first." >&2
    exit 1
fi
echo "Using gene panel: ${GENE_PANEL}"

python3 ~/hest-vae-paper/train_vae_paper.py \
    --data-dir ${DATASET_DST} \
    --gene-panel ${GENE_PANEL} \
    --output-dir ${EXP_DIR} \
    --fold 0 \
    "$@"

cp -a ${EXP_DIR}/. ${PERSIST_DIR}/
