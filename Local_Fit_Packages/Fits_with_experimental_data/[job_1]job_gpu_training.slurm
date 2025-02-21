#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 8:00:00
#SBATCH -J LocalFit_job_training
#SBATCH -o tuner.out
#SBATCH -e tuner.err
#SBATCH --mem=256000
#SBATCH --array=1-100

module purge
module load apptainer tensorflow/2.13.0

apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif [step_1]Local_Fit_Per_KinematicSet_by_User.py $SLURM_ARRAY_TASK_ID
