#!/bin/bash
#SBATCH --partition=standard
#SBATCH -c 4
#SBATCH -t 00:10:00
#SBATCH -J opposite_autoencoder_1-50_plots
#SBATCH -o gen_plots.out
#SBATCH -e gen_plots.err
#SBATCH --mem=256000

module purge
module load gcc/12.4.0 miniforge/24.3.0-py3.11

python [step_4]Gen_plots.py
python [step_5]find_low_cffs.py