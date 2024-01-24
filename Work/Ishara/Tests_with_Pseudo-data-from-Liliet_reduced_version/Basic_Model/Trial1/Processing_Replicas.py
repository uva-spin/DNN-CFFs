###################################
##  Written by Ishara Fernando   ##
##  Revised Date: 01/24/2024     ##
###################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px  # Import plotly express

original_data_file = 'pseudo_basic_BKM10_Jlab_all_t2.csv'

def CollectingColumns(folder_path):
    tempk = []
    tempQQ = []
    tempx = []
    tempt = []
    tempF = []
    tempReH = []
    tempReE = []
    tempReHt = []
    tempdvcs = []
    tempAbsResReH = []
    tempAbsResReE = []
    tempAbsResReHt = []
    tempAbsResdvcs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            tempdf = pd.read_csv(file_path)
            tempk.append(tempdf["k"])
            tempQQ.append(tempdf["QQ"])
            tempx.append(tempdf["x_b"])
            tempt.append(tempdf["t"])
            tempF.append(tempdf["F"])
            tempReH.append(tempdf["ReH"])
            tempReE.append(tempdf["ReE"])
            tempReHt.append(tempdf["ReHt"])
            tempdvcs.append(tempdf["dvcs"])
            tempAbsResReH.append(tempdf["AbsRes_ReH"])
            tempAbsResReE.append(tempdf["AbsRes_ReE"])
            tempAbsResReHt.append(tempdf["AbsRes_ReHt"])
            tempAbsResdvcs.append(tempdf["AbsRes_dvcs"])
    return (np.array(tempk), np.array(tempQQ), np.array(tempx), np.array(tempt), np.array(tempF),
            np.array(tempReH), np.array(tempReE), np.array(tempReHt), np.array(tempdvcs),
            np.array(tempAbsResReH), np.array(tempAbsResReE), np.array(tempAbsResReHt), np.array(tempAbsResdvcs))

def AvgVals(folder_path):
    data_dictionary = {"k": [], "QQ": [], "x_b": [], "t": [], "F": [], "errF": [],
                       "ReH": [], "sigmaReH": [], "ReE": [], "sigmaReE": [],
                       "ReHt": [], "sigmaReHt": [], "dvcs": [], "sigmadvcs": [],
                       "AbsRes_ReH": [], "sigmaAbsRes_ReH": [], "AbsRes_ReE": [], "sigmaAbsRes_ReE": [],
                       "AbsRes_ReHt": [], "sigmaAbsRes_ReHt": [], "AbsRes_dvcs": [], "sigmaAbsRes_dvcs": []}
    results = CollectingColumns(folder_path)
    

    tempF = results[4].mean(axis=0)
    tempFErr = results[4].std(axis=0)
    tempReH = results[5].mean(axis=0)
    tempReHErr = results[5].std(axis=0)
    tempReE = results[6].mean(axis=0)
    tempReEErr = results[6].std(axis=0)
    tempReHt = results[7].mean(axis=0)
    tempReHtErr = results[7].std(axis=0)
    tempdvcs = results[8].mean(axis=0)
    tempdvcsErr = results[8].std(axis=0)
    tempAbsResReH = results[9].mean(axis=0)
    tempAbsResReHErr = results[9].std(axis=0)
    tempAbsResReE = results[10].mean(axis=0)
    tempAbsResReEErr = results[10].std(axis=0)
    tempAbsResReHt = results[11].mean(axis=0)
    tempAbsResReHtErr = results[11].std(axis=0)
    tempAbsResdvcs = results[12].mean(axis=0)
    tempAbsResdvcsErr = results[12].std(axis=0)
    

    data_dictionary["k"] = pd.read_csv(original_data_file, usecols=["k"]).iloc[:len(tempF)]["k"].tolist()
    data_dictionary["QQ"] = pd.read_csv(original_data_file, usecols=["QQ"]).iloc[:len(tempF)]["QQ"].tolist()
    data_dictionary["x_b"] = pd.read_csv(original_data_file, usecols=["x_b"]).iloc[:len(tempF)]["x_b"].tolist()
    data_dictionary["t"] = pd.read_csv(original_data_file, usecols=["t"]).iloc[:len(tempF)]["t"].tolist()
    data_dictionary["F"] = tempF
    data_dictionary["errF"] = tempFErr
    data_dictionary["ReH"] = tempReH
    data_dictionary["sigmaReH"] = tempReHErr
    data_dictionary["ReE"] = tempReE
    data_dictionary["sigmaReE"] = tempReEErr
    data_dictionary["ReHt"] = tempReHt
    data_dictionary["sigmaReHt"] = tempReHtErr
    data_dictionary["dvcs"] = tempdvcs
    data_dictionary["sigmadvcs"] = tempdvcsErr
    data_dictionary["AbsRes_ReH"] = tempAbsResReH
    data_dictionary["sigmaAbsRes_ReH"] = tempAbsResReHErr
    data_dictionary["AbsRes_ReE"] = tempAbsResReE
    data_dictionary["sigmaAbsRes_ReE"] = tempAbsResReEErr
    data_dictionary["AbsRes_ReHt"] = tempAbsResReHt
    data_dictionary["sigmaAbsRes_ReHt"] = tempAbsResReHtErr
    data_dictionary["AbsRes_dvcs"] = tempAbsResdvcs
    data_dictionary["sigmaAbsRes_dvcs"] = tempAbsResdvcsErr
    
    data_dictionary["set"] = pd.read_csv(original_data_file, usecols=["set"]).iloc[:len(tempF)]["set"].tolist()
    data_dictionary["phi_x"] = pd.read_csv(original_data_file, usecols=["phi_x"]).iloc[:len(tempF)]["phi_x"].tolist()

    return pd.DataFrame(data_dictionary)

# Directory containing the CSV files
folder_path = "Replica_Results"

# Use AvgVals function to get the DataFrame with average values
average_df = AvgVals(folder_path)

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


