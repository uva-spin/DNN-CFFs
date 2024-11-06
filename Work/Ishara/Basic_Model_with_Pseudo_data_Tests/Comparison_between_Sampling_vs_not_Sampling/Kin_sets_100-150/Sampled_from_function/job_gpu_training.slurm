#!/bin/bash
#SBATCH -A spinquest
#SBATCH -p gpu
#SBATCH --gres=gpu:a100
#SBATCH -c 8
#SBATCH -t 10:00:00
#SBATCH -J LocalFit_Comparison_100-150_Sampled
#SBATCH -o tuner.out
#SBATCH -e tuner.err
#SBATCH --mem=256000
#SBATCH --array=1-300

module purge
module load apptainer tensorflow/2.13.0

apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Pseudo_Basic_Local_Fit_Per_KinematicSet_by_User.py $SLURM_ARRAY_TASK_ID
