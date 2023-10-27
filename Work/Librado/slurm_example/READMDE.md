# Using Slurm File and job script


A brief description of what this project does and its purpose.

## Prerequisites

- Make sure you have the required Python environment and libraries set up. If you're using `conda`, ensure the environment is activated before submitting jobs.
- SLURM should be configured and available for job submission.

## Usage

This project allows you to run multiple jobs concurrently using SLURM and the provided script. The main components are:

- `generate_pseudodata.py`: The Python script that does the actual data generation.
- `generate_data.slurm`: The SLURM batch script that sets up the job parameters and runs the Python script.
- `njobs.sh`: A shell script that submits multiple jobs at once to SLURM.

### Running Jobs

1. If you wish to run a single job, you can use:

   ```bash
   sbatch generate_data.slurm [PYTHON_SCRIPT_NAME]
```

2. To run multiple jobs at once:

  ```bash
 ./njobs.sh [PYTHON_SCRIPT_NAME] [NUMBER_OF_JOBS]
```

for instance, to run 'generate_pseudodata.py' for 3 jobs you'd execute:

```bash
./njobs.sh generate_pseudodata.py 3

```

# Outputs

The outputs of each job will be in the `Output` folder, with each job getting a different filename based on the order it was run 1 - n, n being your inputted number of jobs

There will also be to new files, `slurm.err` and slurm.out` the .err file is useful to see debugging information if an error occurs, the .out file shows what would be printed to terminal in each job

# Troubleshooting

If you have issues hen running njobs.sh, ensure it is executable:
```bash
chmod +x njobs.sh
```


