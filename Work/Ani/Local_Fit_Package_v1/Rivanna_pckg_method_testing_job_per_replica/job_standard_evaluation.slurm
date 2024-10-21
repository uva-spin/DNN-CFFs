#!/bin/bash
#SBATCH --partition=standard
#SBATCH -A spinquest
#SBATCH -c 4

# Purge any previously loaded modules
module purge

# Load the necessary module for TensorFlow
module load apptainer tensorflow/2.13.0

# Run your Python script with apptainer and the container
apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Evaluation.py
