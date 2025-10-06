#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 2:00:00
#SBATCH -J LocalFit_job_gen_csvs
#SBATCH -o evaluation.out
#SBATCH -e evaluation.err
#SBATCH --mem=256000

module purge
module load apptainer tensorflow/2.13.0

apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif [step_2]Gen_csvs.py