#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 1:00:00
#SBATCH -J Gen_pseudodata_set55_1000pts
#SBATCH -o gen_pseudodata.out
#SBATCH -e gen_pseudodata.err
#SBATCH --mem=256000

module purge
module load apptainer tensorflow/2.17.0

apptainer run --nv $CONTAINERDIR/tensorflow-2.17.0.sif [step_1]Generate_psuedo_data_with_sampling.py
