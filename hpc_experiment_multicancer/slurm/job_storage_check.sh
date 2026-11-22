#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --time=00:15:00

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdg

echo "=== /var/local/storage/${USER} (this node, obiwan/hpc01) ==="
du -sh /var/local/storage/${USER} 2>/dev/null || echo "(empty or missing)"
du -sh /var/local/storage/${USER}/* 2>/dev/null || true

echo ""
echo "=== /opt/storage/datasets/HEST total + breakdown ==="
du -sh /opt/storage/datasets/HEST
du -sh /opt/storage/datasets/HEST/*

echo ""
echo "=== /opt/storage overall free space ==="
df -h /opt/storage

echo ""
echo "=== Candidate audit: wsis/*.tif and st/*.h5ad not referenced by selected_wsis.csv ==="
python3 -c "
import os
import pandas as pd

df = pd.read_csv('/opt/storage/datasets/HEST/selected_wsis.csv')
selected_ids = set(df['id'])

wsis_dir = '/opt/storage/datasets/HEST/wsis'
st_dir = '/opt/storage/datasets/HEST/st'
patches_dir = '/opt/storage/datasets/HEST/patches_multicancer'

def unused(d, ext):
    if not os.path.isdir(d):
        return []
    files = [f[:-len(ext)] for f in os.listdir(d) if f.endswith(ext)]
    return sorted(set(files) - selected_ids)

print('Unused wsis/*.tif (not in selected_wsis.csv):', unused(wsis_dir, '.tif'))
print('Unused st/*.h5ad (not in selected_wsis.csv):', unused(st_dir, '.h5ad'))
print('Unused patches_multicancer/*.h5 (not in selected_wsis.csv):', unused(patches_dir, '.h5'))

# Sizes of the unused ones
for d, ext, label in [(wsis_dir, '.tif', 'wsis'), (st_dir, '.h5ad', 'st')]:
    for fid in unused(d, ext):
        p = os.path.join(d, fid + ext)
        sz = os.path.getsize(p) / 1e9
        print(f'  {label}/{fid}{ext}: {sz:.2f} GB')
"
