# Interactive Plot Script

## Overview
- This script provides an interactive plotting experience, allowing users to view and manipulate both 2D and 3D plots of various datasets.
- It's designed to read CSV files from a specific folder, average the results, and display them in interactive 2D or 3D plots.

## Prerequisites
- Python installed on your system along with several libraries.
- The script cannot be run on systems like Rivanna that only offer terminal access, as it requires a graphical interface for interactive plotting.

## Installation
- Ensure you have the following libraries installed:
  - matplotlib - For plotting graphs.
  - pandas - For data manipulation and analysis.
  - numpy - For numerical computations.
  - plotly - For creating interactive plots.
- Install these libraries using pip:

```bash
pip install matplotlib pandas numpy plotly
```
## Usage
1. **Prepare Your Data:** Place your CSV files in the `../DNNvalues folder`. The script will average the data from these files.
2. **Run the Script:** Execute the script from a command line or an IDE that supports Python scripts. Use the following command format:
	- For 3D Plot: `python plot_interactive.py 3d <y_column> <z_column>`
	- For 2D Plot: `python plot_interactive.py 1d <y_column> <z_column>`
	- Available columns: 'Bin,' 'F', 'ReH', 'ReHt', 'dvcs.'
3. **Interactive Plotting:** Once the script is running, it will display the plot. In 3D plots, you can rotate, zoom, and shift perspective. In 2D plots, you can pan, zoom, and scale the plot interactively.

## Note for Rivanna Users
- As Rivanna primarily offers terminal access and does not support graphical interfaces required for interactive plotting, this script cannot be run directly on Rivanna.
- To use this script, run it on a local machine or any environment where you have access to a graphical user interface.