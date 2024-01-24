# Sample Codes to Get Started

There are three directories/folders in this folder.


## An Example for generating pseudo-data

   In this folder called "Example_PseudoData_Generation", you can find a file called 'Generate_Pseudo_Data_Basic_Model.py'.
   This contains the so-called "basic model" for CFFs with the generic form of
   $$CFF(x_B,t) = (a x_B^2 + b x_B) e^{ct^2 + dt + e}+f $$

   The following parameters are used as a starting point:
   | --- | a | b | c | d | e | f |
   | --- | --- | --- | --- | --- | --- | --- |
   | ReH | -4.41 | 1.68 | -9.14 |-3.57 | 1.54 | -1.37 |
   | ReE | 144.56 | 149.99 | 0.32 | -1.09 | -148.49 | -0.31 |
   | ReHe |  -1.86 | 1.50 | -0.29 | -1.33 | 0.46 | -0.98 |
   | DVCS | 0.50  | -0.41 | 0.05 | -0.25 | 0.55 | 0.166 |

The relevent scripts can be found in the "Example_PseudoData_Generation" folder.

## An Example for Local-Multivariate-Inference (LMI)

A multivariate fit is performed in $Q^2$, $t$, and $x_b$ using a DNN at many fixed kinematics across the independent variable $\phi$. 
The analytical fit function (loss function) is defined by the helicity amplitudes so the results can be specific to a particular formalism, similar to the local fit. 
The DNN fit incorporates all of the information across the phase space of the experimental data resulting in a model that can interpolate and extrapolate. 
With this approach, there is no preconceived analytical expression defined, so there are no initial biases to contend with.

Referece: https://confluence.its.virginia.edu/display/twist/The+DNN+Extraction+Approach

The relevent scripts can be found in the "Example_LMIFit_Code" folder.
   

## An Example for Local-Fit
The Compton form factors (CFFs) are extracted through a fit at fixed kinematics, usually across the independent variable $\phi$, the azimuthal angle between the lepton and hadron scattering planes. 
The fit independently determines the CFFs from measurements between different fixed kinematic bins. 
The analytical fit function (or loss function for ANNs) is defined by the helicity amplitudes so the results can be specific to a particular formalism. 
Without the necessary constraints using multiple observables in simultaneous fitting, there is a lack of uniqueness leading to large systematic errors in the extraction.

Referece: https://confluence.its.virginia.edu/display/twist/The+DNN+Extraction+Approach

The relevent scripts can be found in the "Example_LocalFit_Code" folder.
