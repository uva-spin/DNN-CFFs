###################################
##  Written by Ishara Fernando   ##
##  Revised Date: 01/25/2024     ##
###################################

import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV
df = pd.read_csv('average_values_from_replicas.csv')

# Drop duplicate rows based on 'set' column
df_unique = df.drop_duplicates(subset='set')

# Create a folder named 'CFFs_Histograms' if it doesn't exist
output_folder = 'CFFs_Histograms'
os.makedirs(output_folder, exist_ok=True)

# Plotting ReH values with error bars over 'x_b'
plt.figure(figsize=(10, 6))
plt.errorbar(df_unique['x_b'], df_unique['ReH'], yerr=df_unique['sigmaReH'], fmt='o', label='ReH values')
plt.xlabel('x_b')
plt.ylabel('ReH values')
plt.title('ReH values with error bars over x_b')
plt.legend()
plt.grid(True)

# Save the plot as a PDF inside the 'CFFs_Histograms' folder
output_path_x = os.path.join(output_folder, 'ReH_over_x_b.pdf')
plt.savefig(output_path_x, format='pdf')
plt.show()

# Plotting ReH values with error bars over 't'
plt.figure(figsize=(10, 6))
plt.errorbar(df_unique['t'], df_unique['ReH'], yerr=df_unique['sigmaReH'], fmt='o', label='ReH values')
plt.xlabel('t')
plt.ylabel('ReH values')
plt.title('ReH values with error bars over t')
plt.legend()
plt.grid(True)

# Save the plot as a PDF inside the 'CFFs_Histograms' folder
output_path_t = os.path.join(output_folder, 'ReH_over_t.pdf')
plt.savefig(output_path_t, format='pdf')
plt.show()

print(f'Plots saved in {output_folder} folder as ReH_over_x_b.pdf and ReH_over_t.pdf')

