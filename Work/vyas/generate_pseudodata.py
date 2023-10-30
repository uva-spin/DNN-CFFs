import pandas as pd
import numpy as np

from non_model_utils import F1F2
from model_utils import F_calc

file_name = 'data.csv'

df = pd.read_csv(file_name, dtype=np.float32)
df.set_index('index')

fns = F1F2()
calc = F_calc()

interval = 18

for i in range(len(df)):
  row = df.loc[i]
  kin_ranges = [] # QQ, x_b, t, k
  for kin in [row['QQ'], row['x_b'], row['t'], row['k']]:
    low = kin * .95
    high = kin * 1.05
    kin_ranges.append(np.linspace(min(low, high), max(low, high), 3, dtype=np.float32))
  rows = []
  for phi in range(interval, 361, interval): # starts at interval, ends at 360 (inclusive)
    f_calcs = []
    for QQ in kin_ranges[0]:
      for x_b in kin_ranges[1]:
        for t in kin_ranges[2]:
          F1, F2 = fns.f1_f21(t)
          for k in kin_ranges[3]:
            f_calcs.append(calc.fn_1([phi, QQ, x_b, t, k, F1, F2], [row['ReH'], row['ReE'], row['ReHtilde'], row['dvcs']]))
    f_calcs = np.array(f_calcs)
    rows.append([int(row['index']), row['k'], row['QQ'], row['x_b'], row['t'], phi, np.mean(f_calcs), np.std(f_calcs), row['ReH'], row['ReE'], row['ReHtilde'], row['dvcs']])
    print(i, phi)

  new_df = pd.DataFrame(rows, columns=['experiment', 'k', 'QQ', 'x_b', 't', 'phi_x', 'F', 'sigmaF', 'ReH', 'ReE', 'ReHtilde', 'dvcs'])

  new_df.to_csv(f'BKM_pseudodata_{i}.csv', index_label='index')