#!/bin/bash
#SBATCH -A spinquest
#SBATCH -p gpu
#SBATCH --gres=gpu:a6000
#SBATCH -c 4
#SBATCH -t 00:45:00  # Reduced time for faster testing
#SBATCH -J test_localfit_arc_02  # New job name for the smaller test
#SBATCH -o output_test_%A_%a.out  # Output for each array task
#SBATCH -e output_test_%A_%a.err  # Error for each array task
#SBATCH --mem=64000  # Reduced memory
#SBATCH --array=1-4  # Testing for 4 kinematic sets

module purge
module load apptainer tensorflow/2.13.0

# Run the first Python file for each array task
apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif small_test_fit.py $SLURM_ARRAY_TASK_ID

# Check if this is the last task in the array
if [ "$SLURM_ARRAY_TASK_ID" -eq 4 ]; then
    # Wait for all tasks in the array to complete
    scontrol wait $SLURM_JOB_ID
    
    # Now, submit a new job for the second Python script
    sbatch --dependency=afterok:$SLURM_JOB_ID <<EOF
#!/bin/bash
#SBATCH -A spinquest
#SBATCH -p gpu
#SBATCH --gres=gpu:a6000
#SBATCH -c 4
#SBATCH -t 00:30:00  # Adjust as needed for the second script
#SBATCH -J violin_plot_generation
#SBATCH -o output_violin_plots.out
#SBATCH -e output_violin_plots.err
#SBATCH --mem=32000  # Adjust as needed for the violin plot script

module purge
module load apptainer tensorflow/2.13.0

# Run the second Python file for generating violin plots
apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif generate_violin_plots.py
EOF
fi
