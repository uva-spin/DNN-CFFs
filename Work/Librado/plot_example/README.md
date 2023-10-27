# `comparison.py` - A Tool for Comparing Expected vs. True Results



This Python script, `comparison.py`, facilitates the creation of comparison plots between expected and true result values. The primary objective is to visualize deviations, similarities, and performance metrics between the analyzed datasets.



## Table of Contents

- [Prerequisites](#prerequisites)

- [Usage](#usage)

- [Options](#options)

- [Example](#example)

- [Notes](#notes)



## Prerequisites



Ensure you have the following prerequisites:

- Python (recommended version: 3.x)

- Required Python libraries: 

  - argparse

  - numpy

  - pandas

  - matplotlib

  - os

  - scipy



## Usage



To use the script, navigate to the directory containing `comparison.py` on Rivanna and execute the following command:



$ python comparison.py [CFF_NAME] [OPTIONS]





Replace `[CFF_NAME]` with the name of the Compton Form Factor (CFF) you want to plot, and `[OPTIONS]` with any optional arguments described below.



## Options



Here are the command-line arguments you can use:



- `cff` : Name of the CFF to plot.

- `--true_color` : Color for true values. Default is red (`r`).

- `--fit_color` : Color for fitted values. Default is blue (`b`).

- `--point_size` : Size of the data points. Default is `5`.

- `--hide_error_bars` : If present, error bars will not be shown.

- `--hide_mean_points` : If present, mean points will not be shown.

- `--save_path` : Path to save the plot image.

- `--hide_plot` : If present, the plot will not be displayed.

- `--y_range` : Range for the y-axis (format: min,max).

- `--x_range` : Range for the x-axis (format: min,max).

- `--folder_path` : Path to the folder containing the CSV files.

- `--file_name` : Name of the output file. Default is `comparison_plot`.

- `--true_value_path` : Path to the CSV file containing true values.



## Example



Here's a sample command:



$ python comparison.py cff_name --folder_path /path/to/folder --y_range 0,10 --x_range 1,100 --save_path /path/to/save --hide_mean_points --hide_plot





This example plots a comparison graph for a specific CFF using the provided options.



## Notes



- All necessary files/folders should be present in the directory where this README resides.

- If you encounter any issues or need further explanations, refer to the source code comments or reach out to the repository maintainers.



---



Try to run




