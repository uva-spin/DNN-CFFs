#!/bin/bash
#SBATCH -A spinquest
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 24
#SBATCH --partition=standard
#SBATCH -t 72:00:00
#SBATCH -J Localfit_standard
#SBATCH -o terminal.out
#SBATCH -e terminal.err
#SBATCH --array=1-196

source /home/lba9wf/miniconda3/etc/profile.d/conda.sh
conda activate env

python3 Pseudo_Basic_Local_Fit_User_Input_Kinematic.py $SLURM_ARRAY_TASK_ID