# Interactive Plot Script

## Overview
- This script provides an interactive 3D plotting experience, allowing users to view and manipulate 3D plots of various data sets.
- It's designed to read CSV files from a specific folder, average the results, and display them in an interactive 3D plot.

## Prerequisites
- To use this script, you'll need Python installed on your system along with several libraries.
- The script cannot be run on systems like Rivanna that only offer terminal access, as it requires a graphical interface for the interactive plot.

## Installation
- Ensure you have the following libraries installed:
  - matplotlib - For plotting graphs.
  - pandas - For data manipulation and analysis.
  - numpy - For numerical computations.
- Install these libraries using pip:

bash ```
pip install matplotlib pandas numpy
```

## Usage
1. **Prepare Your Data:** Place your CSV files in the `../DNNvalues` folder. The script will average the data from these files.
2. **Run the Script:** Execute the script from a command line or an IDE that supports Python scripts.
3. **Interactive Plotting:** Once the script is running, it will display a 3D plot. You can interact with this plot by rotating, zooming, and shifting perspective.

## Note for Rivanna Users
- As Rivanna primarily offers terminal access and does not support graphical interfaces required for interactive plotting, this script cannot be run directly on Rivanna.
- Run this script on a local machine or any environment where you have access to a graphical user interface.
