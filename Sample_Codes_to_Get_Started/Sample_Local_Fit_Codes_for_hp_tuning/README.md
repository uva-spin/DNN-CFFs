There are two directories/folders in this folder.

   The sample (pseudo) data in this folder uses the so-called "basic model" for CFFs with the generic form of
   $$CFF(x_B,t) = (a x_B^2 + b x_B) e^{ct^2 + dt + e}+f $$

   The following parameters are used as a starting point:
   | --- | a | b | c | d | e | f |
   | --- | --- | --- | --- | --- | --- | --- |
   | ReH | -4.41 | 1.68 | -9.14 |-3.57 | 1.54 | -1.37 |
   | ReE | 144.56 | 149.99 | 0.32 | -1.09 | -148.49 | -0.31 |
   | ReHe |  -1.86 | 1.50 | -0.29 | -1.33 | 0.46 | -0.98 |
   | DVCS | 0.50  | -0.41 | 0.05 | -0.25 | 0.55 | 0.166 |

The relevant scripts can be found in the "Example_PseudoData_Generation" folder.

## Keras_Tuner_Example

There is an example code Keras_Tuner_Pseudo_Basic_Local_Fit.py which performs a tuning for set #1

## Individual_Fit_Example

Once you have the 'best' or 'tuned' hyperparameter configuration, then you can use that in the Pseudo_Basic_Local_Fit.py code, and this code will generate replica models. After running the later mentioned code, use 1D_CFFs_for_Single_Set.py code to evaluate for CFFs.
