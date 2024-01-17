# An Example for Local-Multivariate-Inference (LMI)

A multivariate fit is performed in $Q^2$, $t$, and $x_b$ using a DNN at many fixed kinematics across the independent variable $\phi$. 
The analytical fit function (loss function) is defined by the helicity amplitudes so the results can be specific to a particular formalism, similar to the local fit. 
The DNN fit incorporates all of the information across the phase space of the experimental data resulting in a model that can interpolate and extrapolate. 
With this approach, there is no preconceived analytical expression defined, so there are no initial biases to contend with.

Referece: https://confluence.its.virginia.edu/display/twist/The+DNN+Extraction+Approach

There are four (4) python scripts provided in this folder.

## 1. Sample_LMI_Fit.py

This code is for testing on a local computer or on a single node.

   Run the following commands on your Rivanna terminal to launch the environment, before running the Sample_LMI_Fit.py script.
   ```bash
source /home/lba9wf/miniconda3/etc/profile.d/conda.sh
conda activate env
```

## 2. Sample_LMI_Fit_for_jobs.py

This code is for job submission on Rivanna (or on a Cluster).

   Use the following command to submit your job. Ensure that you have the 'job_LMI.slurm' file in the same directory.
   You can find more details about the 'job_LMI.slurm' file below.
   ```bash
   sbatch job_LMI.slurm
   ```



## 3. BHDVCS_tf_modified.py

This file contains all the relevant definitions from the BKM10 formalism.

## 4. Process_Results.py

This script provides an example of how to use the trained models to generate results. You will have to either add your analysis definitions to this file or create similar files for various analysis purposes using the trained models.

# Other Files

## 5. job_LMI.slurm

This is the job submission file. The most important lines to check are line #10 and #16. 

### line #10
Here you can input the number of replicas you need. In the current version it is set to 10 (because of quick testing purpose).
```bash
#SBATCH --array=0-10
```

### line #16
Here you want to ensure that you are submitting the job with the proper script. Currently it is set to submit the jobs with Sample_LMI_Fit_for_jobs.py
```bash
python3 Sample_LMI_Fit_for_jobs.py $SLURM_ARRAY_TASK_ID
```
