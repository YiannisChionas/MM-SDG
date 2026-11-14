#!/bin/bash
# Multi-job wrapper for job_train.sh. Real measured epoch time on obiwan's
# MIG A100 20GB: 244.2 min/epoch (~2.8x slower than the ~87min/epoch
# estimated from a full RTX 5070 Ti locally -- MIG slices apparently
# restrict compute, not just memory). Only ~1 epoch/job fits even with the
# job time limit raised close to the cluster's 5h max, so this is now
# 1 epoch/job x 12 jobs (was 3x4 before that correction).
#
# Uses afterok (not afterany): if a step fails, later steps are held
# rather than wastefully running on top of a missing/stale checkpoint.

N_STEPS=12
EPOCHS_PER_STEP=1
EXP_NAME="hest_sr_train"
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
        -d afterok:${depends} "${SCRIPT_DIR}/job_train.sh" --epochs ${tot_epochs}
done

echo "Submitted ${N_STEPS} dependent training jobs. Logs: ${LOGDIR}"
