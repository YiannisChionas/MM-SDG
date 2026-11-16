#!/bin/bash
# Fallback multi-job wrapper, only needed if job_train.sh's single-job
# ~2h estimate turns out wrong (same class of surprise as
# hpc_experiment_sr's obiwan-vs-yoda timing mismatch). Uses afterok so a
# failed step holds the rest rather than running against a stale checkpoint.

N_STEPS=3
EPOCHS_PER_STEP=4
EXP_NAME="hest_rna_sr_train"
LOGDIR="/home/$(whoami)/experiments/${EXP_NAME}/logs"
mkdir -p "${LOGDIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

jid=${EXP_NAME}_0
out=$(sbatch -J "$jid" -o "${LOGDIR}/${jid}.out" -e "${LOGDIR}/${jid}.err" \
    "${SCRIPT_DIR}/job_train.sh" --epochs ${EPOCHS_PER_STEP})
echo "$out"
prev=${out##* }

for ((i = 1; i < N_STEPS; i++)); do
    tot_epochs=$(( (i + 1) * EPOCHS_PER_STEP ))
    jid=${EXP_NAME}_$i
    out=$(sbatch -J "$jid" -o "${LOGDIR}/${jid}.out" -e "${LOGDIR}/${jid}.err" \
        -d afterok:${prev} "${SCRIPT_DIR}/job_train.sh" --epochs ${tot_epochs})
    echo "$out"
    prev=${out##* }
done

echo "Submitted ${N_STEPS} dependent training jobs. Logs: ${LOGDIR}"
