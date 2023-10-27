import sys
import os
import pandas as pd
import numpy as np

from non_model_utils import F1F2
from model_utils import F_calc

# Ensure Output directory exists
output_directory = "Output"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# If jobnumber is provided put it in name
if len(sys.argv) > 1:
    job_number = sys.argv[1]
    output_file = f'Output/BKM_pseudodata_{job_number}.csv'
else:
    output_file = 'Output/BKM_pseudodata_default.csv'

file_name = 'data.csv'
df = pd.read_csv(file_name, dtype=np.float32)
df.set_index('index')


fns = F1F2()
calc = F_calc()

df['F1'], df['F2'] = fns.f1_f21(df['t'])

interval = 180
rows = []

job_id = os.getenv('SLURM_ARRAY_TASK_ID', 'default')  # Get SLURM job ID or use 'default'

for i in range(len(df)):
    row = df.loc[i]
    for phi in range(interval, 361, interval):
        F = calc.fn_1([phi, row['QQ'], row['x_b'], row['t'], row['k'], row['F1'], row['F2']], [row['cff1'], row['cff2'], row['cff3'], row['cff4']])
        rows.append([int(row['index']), row['k'], row['QQ'], row['x_b'], row['t'], phi, F, 0, row['F1'], row['F2'], row['dvcs']])
        print(phi, F)

df = pd.DataFrame(rows, columns=['experiment', 'k', 'QQ', 'x_b', 't', 'phi_x', 'F', 'sigmaF', 'F1', 'F2', 'dvcs'])

# Output file to Output/filename.csv
df.to_csv(output_file, index_label='index')
