#!/bin/bash
#SBATCH --partition=yoda
#SBATCH --ntasks 1
#SBATCH --time=04:00:00

# No --gres line: this is a download/IO job, not a GPU job (per the cluster
# guide's instructions for downloading datasets directly on the server).
# Downloads straight to persistent storage -- NOT fast scratch -- since this
# only needs to happen once, not per training job.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

DEST=/opt/storage/users/${USER}/hest-bench
mkdir -p ${DEST}

hf download MahmoodLab/hest-bench --repo-type dataset --local-dir ${DEST}

echo "Download complete: ${DEST}"
du -sh ${DEST}
