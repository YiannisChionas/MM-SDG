#!/bin/bash
# Submits N_STEPS dependent training jobs, each extending the cumulative
# --epochs target by EPOCHS_PER_STEP. train.py auto-resumes from
# checkpoint.pt in --output-dir, so each step only needs a higher target --
# no separate checkpoint-filename bookkeeping needed (unlike the generic
# repo template, which tracks epoch counts in a side file).
#
# Example: want 30 total epochs, ~3h/10-epoch-step -> N_STEPS=3, EPOCHS_PER_STEP=10.
#
# Usage: ./submit_train.sh

N_STEPS=3
EPOCHS_PER_STEP=10
EXP_NAME="hest_cdm_train"
LOGDIR="/home/$(whoami)/experiments/${EXP_NAME}/logs"
mkdir -p "${LOGDIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

jid=${EXP_NAME}_0
sbatch -J "$jid" -o "${LOGDIR}/${jid}.out" -e "${LOGDIR}/${jid}.err" \
    "${SCRIPT_DIR}/job_train.sh" --epochs ${EPOCHS_PER_STEP}

for ((i = 1; i < N_STEPS; i++)); do
    tot_epochs=$(( (i + 1) * EPOCHS_PER_STEP ))
    d=$(( i - 1 ))
    jid=${EXP_NAME}_$i
    depends=$(squeue --noheader --format %i --name ${EXP_NAME}_${d})
    sbatch -J "$jid" -o "${LOGDIR}/${jid}.out" -e "${LOGDIR}/${jid}.err" \
        -d afterany:${depends} "${SCRIPT_DIR}/job_train.sh" --epochs ${tot_epochs}
done

echo "Submitted ${N_STEPS} dependent training jobs. Logs: ${LOGDIR}"
