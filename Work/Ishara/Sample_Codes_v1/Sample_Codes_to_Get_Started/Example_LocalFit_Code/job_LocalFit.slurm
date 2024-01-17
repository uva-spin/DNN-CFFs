#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
# Input the number of replicas in the following line (default is 5)
#SBATCH --array=0-10

# Activating conda environment
source /home/lba9wf/miniconda3/etc/profile.d/conda.sh
conda activate env

python3 Sample_Local_Fit_for_jobs.py $SLURM_ARRAY_TASK_ID
