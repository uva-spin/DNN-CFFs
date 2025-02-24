#!/bin/bash
#SBATCH -A spinquest
#SBATCH -p gpu
#SBATCH --gres=gpu:a100
#SBATCH --constraint=a100_80gb
#SBATCH -c 4
#SBATCH -t 2:00:00
#SBATCH -J pseudo_data_gen_with_sampling
#SBATCH -o tuner.out
#SBATCH -e tuner.err
#SBATCH --mem=256000

module purge
module load apptainer tensorflow/2.13.0

apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif [step_1]Generate_psuedo_data_with_sampling.py