#!/bin/bash
#SBATCH -A spinquest
#SBATCH -p gpu
#SBATCH --gres=gpu:a6000
#SBATCH -c 4
#SBATCH -t 00:45:00  # Reduced time for faster testing
#SBATCH -J test_localfit_arc_02  # New job name for the smaller test
#SBATCH -o output_test.out
#SBATCH -e output_test.err
#SBATCH --mem=64000  # Reduced memory
#SBATCH --array=1-4  # Testing for 4 kinematic sets

module purge
module load apptainer tensorflow/2.13.0

apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif small_test_fit.py $SLURM_ARRAY_TASK_ID
