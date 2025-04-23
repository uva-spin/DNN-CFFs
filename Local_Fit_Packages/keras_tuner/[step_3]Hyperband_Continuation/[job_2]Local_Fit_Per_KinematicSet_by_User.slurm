#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 6:00:00
#SBATCH -J TopModels_Fit
#SBATCH -o training.out
#SBATCH -e training.err
#SBATCH --mem=256000
#SBATCH --array=1-50

module purge
module load apptainer tensorflow/2.13.0

apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif [step_2]Local_Fit_Per_KinematicSet_by_User.py $SLURM_ARRAY_TASK_ID
