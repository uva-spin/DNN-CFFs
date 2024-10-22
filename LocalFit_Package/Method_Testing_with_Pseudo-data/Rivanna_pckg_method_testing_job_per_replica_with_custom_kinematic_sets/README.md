# Important Information

## Files that user must modify/check before submitting jobs in Rivanna

### 1. Pseudo_Basic_Local_Fit_Per_KinematicSet_by_User.py

1. Line #40: Data file: Ensure that you have the data file name properly with the proper path
2. Line #45: scratch_path: Ensure to provide 'your' UVA_ID in this path
3. Line #60: kinematic_sets: Ensure to provide which kinematic sets that you need to perform the fit
4. Lines 49-54: Hyperparameters

### 2. Update job_gpu_training.slurm

1. Line #11: #SBATCH --array=1-10: By default it is set to replica IDs ranging from 1 to 10. If you want to submit 10 jobs per kinematic set (where each job is handling one replica), then you can copy this slurm file and modify this line with the range you want. 
