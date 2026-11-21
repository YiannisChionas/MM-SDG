#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

python3 ~/git/MM-SDG/hpc_experiment_multicancer/_test_tifffile.py
