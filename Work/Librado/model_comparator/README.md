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

bash ```
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
