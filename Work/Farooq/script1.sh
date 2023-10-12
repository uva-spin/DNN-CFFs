#!/bin/bash

# Initialize conda for the Bash shell (if not already initialized)
# Make sure to replace 'your-conda-environment' with the name of your conda environment
conda init bash
source ~/.bashrc

# Activate your conda environment
conda activate tf_environment

# Change to the directory where your Python script is located
cd code_playground/

# Run your Python script (replace 'script.py' with your script name)
python code1.py

# Deactivate the conda environment (optional)
conda deactivate
