import pandas as pd
import numpy as np

from non_model_utils import F1F2
from model_utils import F_calc

file_name = 'data.csv'

df = pd.read_csv(file_name, dtype=np.float32)
df.set_index('index')

fns = F1F2()
calc = F_calc()

df['F1'], df['F2'] = fns.f1_f21(df['t'])

interval = 180

rows = []

for i in range(len(df)):
  row = df.loc[i]
  for phi in range(interval, 361, interval): # starts at interval, ends at 360 (inclusive)
    F = calc.fn_1([phi, row['QQ'], row['x_b'], row['t'], row['k'], row['F1'], row['F2']], [row['cff1'], row['cff2'], row['cff3'], row['cff4']])
    rows.append([int(row['index']), row['k'], row['QQ'], row['x_b'], row['t'], phi, F, 0, row['F1'], row['F2'], row['dvcs']])
    print(phi, F)

df = pd.DataFrame(rows, columns=['experiment', 'k', 'QQ', 'x_b', 't', 'phi_x', 'F', 'sigmaF', 'F1', 'F2', 'dvcs'])

df.to_csv('BKM_pseudodata.csv', index_label='index')