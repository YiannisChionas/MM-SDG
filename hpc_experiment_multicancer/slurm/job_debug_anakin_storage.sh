#!/bin/bash
#SBATCH --partition=anakin
#SBATCH --ntasks 1
#SBATCH --time=00:05:00

whoami
id
echo "---"
ls -la /var/local/storage/ 2>&1
echo "---"
ls -la /var/local/storage/${USER} 2>&1
echo "---"
df -h /var/local/storage 2>&1
echo "---"
mount | grep "var/local" 2>&1
