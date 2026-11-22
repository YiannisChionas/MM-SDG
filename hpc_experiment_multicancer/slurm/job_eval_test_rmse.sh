#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

python3 ~/git/MM-SDG/hpc_experiment_multicancer/_eval_test_rmse.py
