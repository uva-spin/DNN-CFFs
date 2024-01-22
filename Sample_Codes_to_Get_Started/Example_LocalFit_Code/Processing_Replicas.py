###################################
##  Written by Ishara Fernando   ##
##  Revised Date: 01/22/2024     ##
###################################

import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px  # Import plotly express

# Directory containing the CSV files
folder_path = "Replica_Results"

# Dictionary to store the average values for each column
average_values = {col: [] for col in ["ReH", "ReE", "ReHt", "dvcs", "AbsRes_ReH", "AbsRes_ReE", "AbsRes_ReHt", "AbsRes_dvcs"]}

# List to store the columns needed for the scatter plot
scatter_columns = ['x_b', 'k', 't', 'QQ']

# Iterate over each CSV file in the directory
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Calculate the average for each row for all columns
        for col in average_values.keys():
            # Extract the numeric values from the column
            values = pd.to_numeric(df[col], errors='coerce')
            # Calculate the row-wise average and append to the list
            average_values[col].append(values.mean(skipna=True))

# Create a DataFrame with the average values
average_df = pd.DataFrame(average_values)

# Add 'x_b', 'k', 't', 'QQ' columns to average_df
average_df[scatter_columns] = pd.read_csv(os.path.join(folder_path, os.listdir(folder_path)[0]), usecols=scatter_columns).iloc[:len(average_df)]

# Save the DataFrame to a CSV file
average_df.to_csv("average_values_from_replicas.csv", index=False)

# Plotting subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('1-D Histograms of Average Values')

# Plot histograms for each column
for i, col in enumerate(["AbsRes_ReH", "AbsRes_ReE", "AbsRes_ReHt", "AbsRes_dvcs"]):
    data = average_df[col]
    ax = axs[i // 2, i % 2]
    ax.hist(data, bins=30, edgecolor='black')
    ax.set_title(f'{col} Average')
    ax.set_xlabel('Average Value')
    ax.set_ylabel('Frequency')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for suptitle

# Save the plot as a PDF file
plt.savefig("average_histograms.pdf")

# Function to create 4D scatter plot using average_df
def create_4D_scatter_plot_HTML(df, acc_column, acc_range, title, filename):
    filtered_df = df[(df[acc_column] > acc_range[0]) & (df[acc_column] <= acc_range[1])]
    fig = px.scatter_3d(filtered_df, x='x_b', y='QQ', z='t', color=acc_column, title=title, color_continuous_scale='viridis')
    fig.update_layout(scene=dict(xaxis_title='x_b', yaxis_title='QQ', zaxis_title='t'))
    fig.write_html(filename)

# Use create_4D_scatter_plot_HTML for each residual column
create_4D_scatter_plot_HTML(average_df, 'AbsRes_ReH', (0, 1), 'ReH Residuals', 'kinematics_ReH_Residuals.html')
create_4D_scatter_plot_HTML(average_df, 'AbsRes_ReE', (0, 1), 'ReE Residuals', 'kinematics_ReE_Residuals.html')
create_4D_scatter_plot_HTML(average_df, 'AbsRes_ReHt', (0, 1), 'ReHt Residuals', 'kinematics_ReHt_Residuals.html')
create_4D_scatter_plot_HTML(average_df, 'AbsRes_dvcs', (0, 1), 'dvcs Residuals', 'kinematics_dvcs_Residuals.html')

