#!/bin/bash
#SBATCH -A ishara
#SBATCH -p gpu
#SBATCH --gres=gpu:a100
#SBATCH --constraint=a100_80gb
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J tunertest_selected_set
#SBATCH -o tuner.out
#SBATCH -e tuner.err
#SBATCH --mem=256000
#SBATCH --array=1-5

module purge
module load apptainer tensorflow/2.13.0

apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Pseudo_Basic_Local_Fit_User_Input_Kinematic.py $SLURM_ARRAY_TASK_ID