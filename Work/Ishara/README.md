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
    0. Creates folders for 'DNNmodels', 'Losses_CSVs', 'Losses_Plots', 'Replica_Results', 'Comparison_Plots'.
    1. Sampling total-cross-section $F$ with its uncertainty: generating replicas
       It's implemented as a 'For' loop in this example. You can simply modify it in order to submit parallel jobs on Rivanna.
    2. Producing training loss and validation loss.
    3. Producing models with .h5 extension for each replica.
    4. Producing comparison plots for CFFs ($ReH$, $ReE$, $Re\\tilde{H}$, $dvcs$), and total cross-section F (just a single value: this needs to be modified to plot F vs phi plot for each kinematic range.)

   ## Things to pay attention in the LMIFit.code (the current version)
   
   1. Number of replicas set to 2 (only for demonstrational purpose)
   2. Number of EPOCHS set to 5 (only for demonstrational purpose)
   3. DNNmodel is definded in the section starting frm line #84
   4. If you add layers to the DNNmodel, ensure to modify line #158 and #158 with the correct layer to generate comparison plots.
   5. Check the phi binninng in the 'PseudoData_from_the_Basic_Model.csv' and make sure the binning is compatible with the line #182, #183 in the LMIFit.py code.

  ## To Do List
Create a dedicated folder which contain the following scripts that loads the models saved in 'LMIFit/DNNmodels/' directory.
1. Generate mean and standard deviation for each CFFs for each kinematic set
2. Generate predicted F vs phi and making plots for each kinematic set (including the 'true' values of F vs phi)
3. Evaluate the 'accuracy' and 'precision' of each CFF
4. 1D and 2D visualization of the prdicted CFFs against the 'true' CFFs
