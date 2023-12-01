# README for TensorFlow Model Inspection Tool

## Overview
This tool allows users to interactively inspect TensorFlow model files (.h5). It provides functionalities such as displaying the model summary, plotting the model architecture, and showing information about layers and parameters. You can run this tool without additional environment, just if you want to see cute graphics then install the environmnt. If you are running tensorflow on personal computer, it is not currently supported on python 3.12 so you will have to downgrade python.

## Prerequisites
- Python environment with TensorFlow installed.
- Access to a terminal or command-line interface.
- The TensorFlow model file (.h5) to inspect.

## Environment Setup
To set up the required environment, especially if you're using Rivanna, follow these steps:

1. **Install Miniconda or Anaconda**: If you haven't already, install Miniconda or Anaconda to manage Python environments.

2. **Clone or Download the Required Files**: Ensure you have the `environment_1.1.yml` file and the TensorFlow model file (.h5) ready in your workspace.

3. **Create and Activate the Environment**:
   - Navigate to the directory containing `environment_1.1.yml`.
   - Run the following command to create a new environment:
     ```bash
     conda env create -f environment_1.1.yml
     ```
   - Once the environment is created, activate it using:
     ```bash
     conda activate environment_1.1
     ```

## Usage
1. **Move the tool**: Place the tool within a folder that contains the BHDVCS_tf_modified file. for example, placing it in model_comparator is fine.
2. **Run the Tool**: Use the following command to start the tool:
   ```bash
   python model_inspection_tool.py path/to/your/model.h5

```