#!/bin/bash
#SBATCH --partition=anakin
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00

set -e  # propagate python/sbatch-step failures -- without this, a crashed
        # job falls through to cp -a below and SLURM reports false success

# First real training run of the multi-cancer paper-scale VAE
# (~220.4M params at the 14,368-gene panel). anakin access for this
# project is temporary (granted for the paper-scale reproduction
# effort) -- do not resubmit here without the user's explicit go-ahead
# each time, per CLAUDE.md Rule #1.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

HEST_DIR=/opt/storage/datasets/HEST
EXP_DIR=/opt/storage/users/${USER}/experiments/hest-vae-multicancer
mkdir -p ${EXP_DIR}

# Stage st/*.h5ad (expression; ~5.6GB) to this node's fast local scratch --
# the VAE only ever reads expression, never patches_multicancer/ (images),
# so that's the only heavy dir worth copying for this job. Avoids repeated
# small-file reads over NFS during build_expr_index, which is suspected of
# having contributed to a stuck/hung job before (SIGTERM during NFS I/O
# wait). selected_wsis.csv and the gene panel are tiny -- read directly
# from NFS, not worth copying.
LOCAL_HEST=/var/local/storage/${USER}/HEST
mkdir -p ${LOCAL_HEST}
if [ ! -d ${LOCAL_HEST}/st ]; then
    cp -a ${HEST_DIR}/st ${LOCAL_HEST}/st
fi

# Sort numerically by the gene-count suffix and take the LARGEST --
# plain `ls | head -1` is alphabetical, so a stale gene_panel_multicancer_0.json
# (or any small number) from an earlier broken run would always win over the
# real, larger panel. Bit us once already.
GENE_PANEL=$(ls ${HEST_DIR}/gene_panel_multicancer_*.json | sort -t_ -k4 -n -r | head -1)
if [ -z "${GENE_PANEL}" ]; then
    echo "No gene_panel_multicancer_*.json found in ${HEST_DIR} -- run job_select_genes.sh first." >&2
    exit 1
fi
echo "Using gene panel: ${GENE_PANEL}"

python3 ~/git/MM-SDG/hpc_experiment_multicancer/train_vae_multicancer.py \
    --hest-dir ${LOCAL_HEST} \
    --selected-wsis-csv ${HEST_DIR}/selected_wsis.csv \
    --gene-panel ${GENE_PANEL} \
    --output-dir ${EXP_DIR} \
    "$@"
