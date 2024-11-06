#!/bin/bash
#SBATCH -A spinquest
#SBATCH -p gpu
#SBATCH --gres=gpu:a100
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -J Evaluation_Comparosn_100-150_Sampled
#SBATCH -o tuner.out
#SBATCH -e tuner.err
#SBATCH --mem=256000

module purge
module load apptainer tensorflow/2.13.0

apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Evaluation.py