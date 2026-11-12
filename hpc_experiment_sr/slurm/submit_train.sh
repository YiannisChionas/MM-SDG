#!/bin/bash
# Multi-job wrapper for job_train.sh. At ~87min/epoch (measured locally,
# extrapolated to cluster A100), 3 epochs/job ~= 4.35h, fitting the 4.75h
# time limit with margin. 4 sequential jobs -> 12 epochs total (~17.4h
# wall-clock across jobs, train_sr.py auto-resumes so this is safe to
# interrupt/requeue at any point).

N_STEPS=4
EPOCHS_PER_STEP=3
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
        -d afterany:${depends} "${SCRIPT_DIR}/job_train.sh" --epochs ${tot_epochs}
done

echo "Submitted ${N_STEPS} dependent training jobs. Logs: ${LOGDIR}"
