#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -e  # propagate python/sbatch-step failures -- without this, a crashed
        # job falls through and SLURM reports false success

# Runs the SAME cancer-type classifier used for hpc_experiment/hpc_experiment_sr
# on the RNA-conditioned experiments' paired samples instead (flat real/
# and synthetic/, all CCRCC) -- a class-conditioning-fidelity number
# directly comparable to eval_classifier.py's CCRCC entry, despite the two
# experiments using completely different conditioning mechanisms.
#
#   sbatch job_eval_classifier_on_rna.sh \
#       --classifier-checkpoint /opt/storage/users/$USER/experiments/hest-classifier/classifier_64.pt \
#       --samples-dir /opt/storage/users/$USER/experiments/hest-rna/samples \
#       --output-json /opt/storage/users/$USER/experiments/hest-classifier/eval_rna_64.json
# or for the 224x224 cascade:
#   sbatch job_eval_classifier_on_rna.sh \
#       --classifier-checkpoint /opt/storage/users/$USER/experiments/hest-classifier/classifier_224.pt \
#       --samples-dir /opt/storage/users/$USER/experiments/hest-rna-sr/samples \
#       --output-json /opt/storage/users/$USER/experiments/hest-classifier/eval_rna_sr_224.json

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

python3 ~/hest-classifier/eval_classifier_on_rna.py \
    --batch-size 64 \
    "$@"
