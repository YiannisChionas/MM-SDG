#!/bin/bash
#SBATCH --partition=obiwan
#SBATCH --ntasks 1
#SBATCH --time=00:10:00

set -e
du -sh /opt/storage/datasets/HEST/patches_multicancer
du -sh /opt/storage/datasets/HEST/st
du -sh /opt/storage/datasets/HEST/selected_wsis.csv /opt/storage/datasets/HEST/gene_panel_multicancer_*.json
