#!/bin/bash
#SBATCH --partition=anakin
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --signal=B:USR1@900

set -e  # propagate python/sbatch-step failures -- without this, a crashed
        # job falls through to cp -a below and SLURM reports false success

# Stage 2 (64x64 + RNA -> 256x256 super-resolution) diffusion UNet, pooled
# multi-cancer. anakin access for this project is temporary (granted for
# the paper-scale reproduction effort) -- do not resubmit here without the
# user's explicit go-ahead each time, per CLAUDE.md Rule #1.
#
# Reads patches_multicancer/*.h5 + st/*.h5ad directly from NFS, same as
# job_train_vae_multicancer.sh / job_train_rna_mc.sh.
#
# Same graceful-shutdown setup as job_train_rna_mc.sh (--signal=B:USR1@900,
# worker_init_fn ignoring the signal in DataLoader workers, and the
# while/wait loop below instead of a single `wait $PID` -- see that
# script's comments for the full debugging history of why each piece is
# needed).

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

HEST_DIR=/opt/storage/datasets/HEST
EXP_DIR=/opt/storage/users/${USER}/experiments/hest-rna-sr-multicancer
mkdir -p ${EXP_DIR}

VAE_CKPT=/opt/storage/users/${USER}/experiments/hest-vae-multicancer/vae_multicancer_best.pt
if [ ! -f ${VAE_CKPT} ]; then
    echo "No VAE checkpoint found at ${VAE_CKPT} -- run job_train_vae_multicancer.sh first." >&2
    exit 1
fi

python3 ~/git/MM-SDG/hpc_experiment_multicancer/train_rna_sr_mc.py \
    --hest-dir ${HEST_DIR} \
    --selected-wsis-csv ${HEST_DIR}/selected_wsis.csv \
    --vae-checkpoint ${VAE_CKPT} \
    --output-dir ${EXP_DIR} \
    "$@" &
PID=$!
trap 'echo "forwarding SIGUSR1 to python (pid $PID)"; kill -USR1 $PID' USR1

RC=0
while kill -0 "$PID" 2>/dev/null; do
    wait "$PID" && RC=0 || RC=$?
done
exit $RC
