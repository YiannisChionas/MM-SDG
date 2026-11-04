#!/bin/bash
# Multi-job wrapper for job_train_rna.sh, same pattern as
# hpc_experiment/slurm/submit_train.sh -- train_rna.py auto-resumes, so each
# step just needs a higher cumulative --epochs target.

N_STEPS=3
EPOCHS_PER_STEP=10
EXP_NAME="hest_rna_train"
LOGDIR="/home/$(whoami)/experiments/${EXP_NAME}/logs"
mkdir -p "${LOGDIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

jid=${EXP_NAME}_0
sbatch -J "$jid" -o "${LOGDIR}/${jid}.out" -e "${LOGDIR}/${jid}.err" \
    "${SCRIPT_DIR}/job_train_rna.sh" --epochs ${EPOCHS_PER_STEP}

for ((i = 1; i < N_STEPS; i++)); do
    tot_epochs=$(( (i + 1) * EPOCHS_PER_STEP ))
    d=$(( i - 1 ))
    jid=${EXP_NAME}_$i
    depends=$(squeue --noheader --format %i --name ${EXP_NAME}_${d})
    sbatch -J "$jid" -o "${LOGDIR}/${jid}.out" -e "${LOGDIR}/${jid}.err" \
        -d afterany:${depends} "${SCRIPT_DIR}/job_train_rna.sh" --epochs ${tot_epochs}
done

echo "Submitted ${N_STEPS} dependent training jobs. Logs: ${LOGDIR}"
