# Local Fit Package

This is contains three folders

Note: The hyper-parameters that are defined in the scripts of these folders can be considered as the current "baseline" which may not be the most up-to-date one.

## Pseudo-data-generator

This folder contains the scripts to generate pseudo-data with and without sampling the generating function. Please check the content inside this folder. 

## Method_Testing_with_Pseudo-data

This folder contains the scripts to perform "method testing" with pseudo-data. Please ensure to copy the relevant pseudo-data file from the "Pseudo-data-generator" folder. There are two sub-folders, each one is different in terms of handling job submission on Rivanna.

## Fits_with_Experimental_Data

This folder contains the scripts to perform the extraction with "Experimental Data". This should be used after the "Method Testing" is done with pseudo-data. It will be an iterative process, therefore you can see the folders are arranged in the structure to be able to use in the iteretive process of extraction and it is upto the "accuracy" and "precision" of the CFFs when to stop the iterations.
