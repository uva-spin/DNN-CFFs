#!/bin/bash

# Python script name
SCRIPT_NAME=$1
# Number of Jobs
NJOBS=$2

# Submit Jobs
for (( i=1; i<=$NJOBS; i++)); do
  sbatch generate_data.slurm $SCRIPT_NAME $i
done
