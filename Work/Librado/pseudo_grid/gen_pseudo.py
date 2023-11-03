import pandas as pd
import numpy as np
import time
import os

from non_model_utils import F1F2
from model_utils import F_calc

# Define a dictionary for the kinematic ranges and bins to make adjustments easier
kinematic_params = {
    'k': {'start': 2.75, 'end': 24, 'bins': 10},
    'QQ': {'start': 1.05443, 'end': 3.47114, 'bins': 10},
    'x_b': {'start': 0.179473, 'end': 0.674604, 'bins': 10},
    't': {'start': -0.599976, 'end': -0.206039, 'bins': 10}
}

# Calculate step sizes based on the ranges and bins
for param, values in kinematic_params.items():
    values['step'] = (values['end'] - values['start']) / values['bins']

# Generate the range arrays
k_range = np.arange(kinematic_params['k']['start'], kinematic_params['k']['end'], kinematic_params['k']['step']).astype(np.float32)
QQ_range = np.arange(kinematic_params['QQ']['start'], kinematic_params['QQ']['end'], kinematic_params['QQ']['step']).astype(np.float32)
x_b_range = np.arange(kinematic_params['x_b']['start'], kinematic_params['x_b']['end'], kinematic_params['x_b']['step']).astype(np.float32)
t_range = np.arange(kinematic_params['t']['start'], kinematic_params['t']['end'], kinematic_params['t']['step']).astype(np.float32)

# Precompute the phi values since they are the same for every iteration
phi_values = np.arange(1, 361, 1).astype(np.float32)  # Assuming interval of 1

cff_values = [1, 1, 1, 1]
experiment_index = 0
sigmaF_value = 0
dvcs_value = 0
fns = F1F2()
calc = F_calc()

# Calculate time it will take to run
total_iterations = len(k_range) * len(QQ_range) * len(x_b_range) * len(t_range) * len(phi_values)
current_iteration = 0
start_time = time.time()

# Function to compute F value
def compute_F(k, QQ, x_b, t, phi, cff_values):
    F1, F2 = fns.f1_f21(t)
    F1, F2 = np.float32(F1), np.float32(F2)
    F = calc.fn_1([phi, QQ, x_b, t, k, F1, F2], cff_values)
    return F, F1, F2

# Pre-allocate memory
num_rows = total_iterations
data = np.empty((num_rows, 11))  # 11 columns
timing_window_start = None
timing_window_end = None

row_index = 0
for k in k_range:
    for QQ in QQ_range:
        for x_b in x_b_range:
            for t in t_range:
                for phi in phi_values:
                    F, F1, F2 = compute_F(k, QQ, x_b, t, phi, cff_values)
                    data[row_index] = [experiment_index, k, QQ, x_b, t, phi, F, sigmaF_value, F1, F2, dvcs_value]
                    experiment_index += 1
                    row_index += 1
                    
                    current_iteration += 1
                    if current_iteration == 1000:
                        timing_window_start = time.time()  # Store the start time at iteration 1000
                        
                    if current_iteration == 2000:
                        timing_window_end = time.time()  # Store the end time at iteration 2000
                        time_for_1000_iterations = timing_window_end - timing_window_start  # Time taken for iterations 1000 to 2000
                        avg_time_per_iteration = time_for_1000_iterations / 1000  # Average time per iteration for iterations 1000 to 2000
                        remaining_iterations = total_iterations - current_iteration
                        estimated_remaining_time = avg_time_per_iteration * remaining_iterations
                        print(f'Estimated remaining time: {estimated_remaining_time} seconds')

                    if current_iteration % 1000 == 0:
                        print(f'Progress: {current_iteration}/{total_iterations} iterations')

# Create DataFrame and save to HDF5
columns=['experiment', 'k', 'QQ', 'x_b', 't', 'phi_x', 'F', 'sigmaF', 'F1', 'F2', 'dvcs']
df = pd.DataFrame(data, columns=columns)
df.to_hdf('BKM_pseudodata_long.h5', key='df', mode='w')
