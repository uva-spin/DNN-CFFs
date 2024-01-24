# An Example for Local-Fit
The Compton form factors (CFFs) are extracted through a fit at fixed kinematics, usually across the independent variable $\phi$, the azimuthal angle between the lepton and hadron scattering planes. 
The fit independently determines the CFFs from measurements between different fixed kinematic bins. 
The analytical fit function (or loss function for ANNs) is defined by the helicity amplitudes so the results can be specific to a particular formalism. 
Without the necessary constraints using multiple observables in simultaneous fitting, there is a lack of uniqueness leading to large systematic errors in the extraction.

Referece: https://confluence.its.virginia.edu/display/twist/The+DNN+Extraction+Approach

## 1. Sample_Local_Fit.py

This code is for testing on a local computer or on a single node.

   Run the following commands on your Rivanna terminal to launch the environment, before running the Sample_LMI_Fit.py script.
   ```bash
source /home/lba9wf/miniconda3/etc/profile.d/conda.sh
conda activate env
```

## 2. Sample_Local_Fit_for_jobs.py

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

## 5. job_LocalFit.slurm

This is the job submission file. The most important lines to check are line #10 and #16. 

### line #10
Here you can input the number of replicas you need. In the current version it is set to 10 (because of quick testing purpose).
```bash
#SBATCH --array=0-10
```

### line #16
Here you want to ensure that you are submitting the job with the proper script. Currently it is set to submit the jobs with Sample_LMI_Fit_for_jobs.py
```bash
python3 Sample_Local_Fit_for_jobs.py $SLURM_ARRAY_TASK_ID
```

