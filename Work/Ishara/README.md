# Framework

## Gen_PseudoData
   In this folder, you can find a file called 'Generate_Pseudo_Data_Basic_Model.py'.
   This contains the so-called "basic model" for CFFs with the generic form of
   $$CFF(x_B,t) = (a x_B^2 + b x_B) e^{ct^2 + dt + e}+f $$

   The following parameters are used as a starting point:
   | --- | a | b | c | d | e | f |
   | --- | --- | --- | --- | --- | --- | --- |
   | ReH | -4.41 | 1.68 | -9.14 |-3.57 | 1.54 | -1.37 |
   | ReE | 144.56 | 149.99 | 0.32 | -1.09 | -148.49 | -0.31 |
   | ReHe |  -1.86 | 1.50 | -0.29 | -1.33 | 0.46 | -0.98 |
   | DVCS | 0.50  | -0.41 | 0.05 | -0.25 | 0.55 | 0.166 |

   Here, I used a kinematics file which is a sliced version from one of the previous pseudodata files from Liliet.
   see https://github.com/extraction-tools/ANN/tree/master/Liliet/PseudoData3

   You can change the angle ($\phi$) binning when you generate pseudodata using the Generate_Pseudo_Data_Basic_Model.py file

   ## LMI Fit

   In this folder, I have used 'BHDVCS_tf_modified.py' which includes all the dependent classes/functions.
   Currently, the 'LMIFit.py' includes the following features 
    1. sampling total-cross-section $F$ with its uncertainty: generating replicas
       It's implemented as a 'For' loop in this example. You can simply modify it in order to submit parallel jobs on Rivanna.
    2. producing training loss and validation loss
    
