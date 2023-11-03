#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --partition=standard
#SBATCH -A spinquest_standard

# Activating conda environment
source /home/lba9wf/miniconda3/etc/profile.d/conda.sh
conda activate env

# Executing Script
python $1 $2
