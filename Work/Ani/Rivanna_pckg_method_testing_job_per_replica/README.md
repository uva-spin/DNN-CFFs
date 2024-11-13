## Important Instructions

The set of 'source' files here are for a single kinematic set. Therefore, the user needs to create copies of the files in this folder into other folder which are created with a meaningful name so the user can easily distinguish which folder is corresponding to which kinematic set.

### Pseudo_Basic_Local_Fit_Per_KinematicSet_by_User.py

Note: The line numbers given in the following steps may be slightly shifted from the orginial (depending on the evolution of the code)

1. Ensure to have a uinique path defined for the defined kinematic set that you want to save the replica models (line #46)
2. Ensure to define j (kienematic set number): By default j=1
3. In the .slurm file that you are going to use, ensure each one has a job name that user can easily distinguish later when user can check the progress 
   line #SBATCH -J LocalFit_job 
