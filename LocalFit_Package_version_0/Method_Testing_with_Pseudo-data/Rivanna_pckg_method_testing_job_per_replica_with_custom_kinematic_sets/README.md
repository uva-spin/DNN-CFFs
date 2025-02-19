# Important information to the user

## Files that user must modify/check before submitting jobs in Rivanna numbered, and the rest is information on the files

### 1.Pseudo_Basic_Local_Fit_Per_KinematicSet_by_User.py

1. Line #40: Data file: Ensure that you have the data file name properly with the proper path
2. Line #45: scratch_path: Ensure to provide 'your' UVA_ID in this path
3. Line #60: kinematic_sets: Ensure to provide which kinematic sets that you need to perform the fit
4. Lines 49-54: Hyperparameters

This code file now includes the creation of F vs phi before training, with random sampling with most points falling within 1 sigma of the standard deviation of F. These files are created under the folder `Cross_Sections_Replicas`. It also allows the user to input kinematics in a list at the start. It is the same as the other training files in all other aspects.

## 2. Update job_gpu_training.slurm

1. Line #11: There are 3 lines of interest here:
2. Ensure the 'path' where you want to save the models is updated. Otherwise, you mayne over-riding the models in an existing folder.

```
--gres=gpu:a100

-t 1:00:00

--array=1-300
```
For the first line, The existing GPUs are the following: \
`v100`, `a40`, `a600`, `a100`, \
so feel free to use any of those.

For the second line, the time, try to keep it within 3 hours. If you need to, modify the array below accordingly.

For the third line, this is the number of replicas. There are two options here. First is to drop the number of replicas so that the first job does replicas 1 to 50, then 51 to 100, etc, or any series of replicas of your choice, but try to keep the time within 3 hours. The other option is to drop the number of kinematic sets so that the jobs don't run too long. Feel free to use a combination of these two so that the jobs are done for a certain number of kinematic sets with a couple replicas at a time, for maximum efficiency when submitting jobs.

Ensure to give a meaningful name for this line: '#SBATCH -J LocalFit_job'

## 3. Evaluation.py

1. Line 51: put your own computing id
2. Ensure to give the 'path' where your models are saved (from the Pseudo_Basic_Local_Fit_Per_KinematicSet_by_User.py), otherwise you may create evaluation results from models from a totally different folder.


This evaluates the trained data over CFFs, evaluates F vs phi, makes a chi-square txt file for how the chi square error on F vs phi, and provides a csv for how well it predicted F vs phi. The F vs phi plots, CFF evaluation plots, and `chi_square.txt` are made in the `Comparison_Plots` folder, while the evaluation of F vs phi is made in the `CFF_Mean_Deviation_Plots` folder. This file takes all available models in the `DNN_CFFs` folder in the scratch path and looks over all of them, and makes models for each available model in that folder as of right now. Be careful of the models currently in that folder before running evaluate.

## 4. job_gpu_evaluation.slurm
```
#SBATCH --gres=gpu:a100
#SBATCH -t 1:00:00
```
These are the only two lines of interest. The gpus that are available on rivanna are mentioned above. For the time, I realize that evaluating depends entirely on the number of replicas and kinematic sets in the `DNN_CFFs` folder. Feel free to increase the time up to 3 hours if necessary. If it takes longer (due to a large number of replicas and kinematic sets), then please contact me and I will have to re-write the part where it reads all existing models in the folder and instead just reads models that the user inputs.

Ensure to give a meaningful name for this line: '#SBATCH -J Evaluation_job'

## 5. Evaluation_csvs_only.py

This file only creates the csvs that would be made in Evaluation.py. The key lines are the same as in Evaluation.py: just make sure the scratch path is correct and the rest will handle itself.

## Evaluation_graphs_from_csvs.py

This file creates the graphs using the csvs. Make sure to set the scratch path to what it originally was in the `Evaluation_csvs_only.py` and make sure to only run this file after making the csvs.

## job_gpu_evaluation_csvs_only.slurm

This is the associated slurm file for `Evaluation_csvs_only.py`. Change the hours if necessary, and feel free to change gpu to a100 only if you prefer.

## job_gpu_evaluation_graphs_from_csvs.slurm

This is the associated slurm file for `Evaluation_graphs_from_csvs.py`, but isn't necessary as this file can just be run locally as well as there is no explicit need for tensorflow.