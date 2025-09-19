#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 20:00:00
#SBATCH -J Gen_csvs_basic_1
#SBATCH -o gen_csvs.out
#SBATCH -e gen_csvs.err
#SBATCH --mem=256000

module purge
module load apptainer tensorflow/2.13.0

apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif [step_3]Gen_csvs.py