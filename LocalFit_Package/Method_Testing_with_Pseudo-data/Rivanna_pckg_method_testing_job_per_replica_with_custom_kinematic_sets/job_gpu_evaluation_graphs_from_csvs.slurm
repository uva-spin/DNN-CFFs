#!/bin/bash
#SBATCH -A spinquest
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 1:00:00
#SBATCH -J LocalFit_job
#SBATCH -o tuner.out
#SBATCH -e tuner.err
#SBATCH --mem=256000

module purge
module load apptainer tensorflow/2.13.0

apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Evaluation_graphs_from_csvs.py