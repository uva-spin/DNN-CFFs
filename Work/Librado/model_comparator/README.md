# Deep Neural Network Model Comparator and Analyzer

## Overview
This script is designed for analyzing and comparing predictions from deep neural network models, specifically tailored for Compton form factors (CFFs) and structure functions (F) predictions. It includes functionalities for 2D and 3D visualization, and performance metrics evaluation.

## Features
- Loads TensorFlow models and makes predictions on given inputs.
- Calculates accuracy and precision of predictions.
- Visualizes data in 2D plots (actual vs. predicted values).
- Generates 3D plots for comparative analysis.
- Saves plots in specified directories for easy access and analysis.

## Prerequisites
- Python environment with necessary libraries installed (TensorFlow, Pandas, NumPy, Matplotlib, etc.).
- Pre-trained TensorFlow models stored in a specific directory.
- Data in CSV format for making predictions.

## Installation and Setup
Ensure you have the following Python libraries installed(if you are running on rivanna, it is included with the env.yml you installed):
- TensorFlow: For running deep neural network models.
- Pandas: For data handling and manipulation.
- NumPy: For numerical computations and array manipulations.
- Matplotlib: For generating plots.

You can install these libraries using pip if not already installed:

```bash
pip install tensorflow pandas numpy matplotlib
```

## Usage
1. **Data Preparation**: Ensure your prediction data is in a CSV file named 'PseudoData_from_the_Basic_Model.csv'.
2. **Model Setup**: Place your TensorFlow models in a folder named 'DNNmodels'. Automatically done when running LMIFit.py
3. **Execution**: Run the script from a Python environment. It will automatically create necessary folders for outputs ('DNN2Dimages', 'DNN3Dimages', 'DNNvalues').
4. **Analysis**: The script will generate 2D and 3D plots for each CFF and structure function F, comparing actual and predicted values. These plots are saved in the designated folders.

## Note
- This script assumes a specific structure for the TensorFlow models and input data. Ensure your models and data conform to the expected format.
- The script must be run in an environment where graphical output is supported, as it generates visual plots.
- To see more interactive and 3d graphs go to InteractivePlots folder

## Additional scripts

### njobs.sh
This script is used to submit multiple jobs simultaneously on systems like Rivanna. It takes to arguments: the name of the python script to be executed and the number of jobs to run.
1. Make sure `njobs.sh` is executable by running `chmod +x njobs.sh`. if that doesn't work try `chmod 755 njobs.sh` or somtimes restarting terminal after command is necessary.
2. Execute the script with the Python script name and the desired number of jobs. For example: `./njobs.sh LMIFIT.py 3` will run th script 3 times.

### generate_data.slurm
This is a SLURM script used to manage job submission on systems like Rivanna. It sets various SLURM job parameters such as the number of tasks, runtime, and partition.
1. For the purpose of this folder `generate_data.slurm` is called by `njobs.sh`. It takes two arguments: the name of the python script and the specific job number(replica number for this folder)
2. You can edit the slurm details for specific job requirements(time, partition, etc.)

