import numpy as np
import pandas as pd

df = pd.read_csv('File_from_Liliet.csv')
sliced = df.loc[df['phi_x'] == 8, :]
df_modified = pd.DataFrame(sliced)
selected_columns = ['k', 'QQ','x_b','t']
df_selected = df_modified[selected_columns]
df_selected.to_csv('Kinemactics.csv', index=False)