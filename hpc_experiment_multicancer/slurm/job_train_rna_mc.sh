#!/bin/bash
#SBATCH --partition=anakin
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --signal=B:USR1@900

set -e  # propagate python/sbatch-step failures -- without this, a crashed
        # job falls through to cp -a below and SLURM reports false success

# Stage 1 (RNA -> 64x64 image) diffusion UNet, pooled multi-cancer. anakin
# access for this project is temporary (granted for the paper-scale
# reproduction effort) -- do not resubmit here without the user's explicit
# go-ahead each time, per CLAUDE.md Rule #1.
#
# Reads patches_multicancer/*.h5 + st/*.h5ad directly from NFS, same as
# job_train_vae_multicancer.sh -- /var/local/storage isn't writable for
# this account on anakin's node (hpc06), unlike obiwan's hpc01.
#
# --signal=B:USR1@900 makes SLURM send SIGUSR1 to this script 900s (15min)
# before the hard time-limit kill. train_rna_mc.py catches it and saves a
# final checkpoint at whatever step it's on, instead of losing everything
# back to the last periodic --checkpoint-every save. The trap below
# forwards the signal from this shell to the python child, since bash
# doesn't do that automatically for a foregrounded command.
#
# First attempt used @300 (5min) and still lost progress -- the job ended
# FAILED with no "checkpointed at step" line ever printed after the signal
# was received, and checkpoint.pt was untouched. Widening to 15 minutes
# gives much more headroom for the in-flight training step (~10 sec/step)
# to finish and the ~2.4GB torch.save() over NFS to complete before the
# real time-limit kill arrives.
#
# Two manual `scancel --signal=USR1 --batch` tests still failed the same
# way within seconds (ruled out a DataLoader-worker-death theory first --
# a worker_init_fn fix for that didn't help either). Root cause: `wait
# $PID` below gets INTERRUPTED by the trapped signal and returns before
# python actually exits, with set -e then killing this whole script
# immediately on that non-zero return -- which makes SLURM tear down the
# job's whole cgroup (SIGKILL included) right as python was starting its
# graceful checkpoint save. The loop below keeps re-issuing `wait` (and
# the `&&`/`||` keeps set -e from firing on an interrupted wait) until
# the process has actually exited.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

HEST_DIR=/opt/storage/datasets/HEST
EXP_DIR=/opt/storage/users/${USER}/experiments/hest-rna-multicancer
mkdir -p ${EXP_DIR}

VAE_CKPT=/opt/storage/users/${USER}/experiments/hest-vae-multicancer/vae_multicancer_best.pt
if [ ! -f ${VAE_CKPT} ]; then
    echo "No VAE checkpoint found at ${VAE_CKPT} -- run job_train_vae_multicancer.sh first." >&2
    exit 1
fi

python3 ~/git/MM-SDG/hpc_experiment_multicancer/train_rna_mc.py \
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
