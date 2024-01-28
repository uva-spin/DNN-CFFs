###################################
##  Written by Ishara Fernando   ##
##  Revised Date: 01/28/2024     ##
###################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px  # Import plotly express

original_data_file = 'pseudo_KM15_BKM10_Jlab_all_t2.csv'

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


def absolute_residual(tr, prd):
    temp_diff = np.abs(tr) - np.abs(prd)
    temp_abs_diff = np.abs(temp_diff) 
    return temp_abs_diff


def CollectingColumnsCFFs(folder_path,avg_df):
    tempDevReH = []
    tempDevReE = []
    tempDevReHt = []
    tempDevdvcs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            tempdf = pd.read_csv(file_path)
            tempDevReH.append(absolute_residual(avg_df["ReH"],tempdf["ReH"]))
            tempDevReE.append(absolute_residual(avg_df["ReE"],tempdf["ReE"]))
            tempDevReHt.append(absolute_residual(avg_df["ReHt"],tempdf["ReHt"]))
            tempDevdvcs.append(absolute_residual(avg_df["dvcs"],tempdf["dvcs"]))
    return (np.array(tempDevReH), np.array(tempDevReE), np.array(tempDevReHt), np.array(tempDevdvcs))



CFFarrays = CollectingColumnsCFFs(folder_path,average_df)


print(len(CFFarrays[0]))
#plt.hist(CFFarrays[0])
#plt.show()

# Function to create histograms and subplots
def plot_histograms(CFFarrays, output_filename="histograms.pdf"):
    plt.figure(figsize=(12, 10))
    plt.suptitle('1D Histograms of CFFs')

    colors = ['blue', 'green', 'orange', 'red']  # Add more colors if needed

    # Plot ReH histogram
    plt.subplot(2, 2, 1)
    for i, dataset in enumerate(CFFarrays[0]):
        plt.hist(dataset, bins=20, color='blue', alpha=0.7)
    plt.title('ReH')

    # Plot ReE histogram
    plt.subplot(2, 2, 2)
    for i, dataset in enumerate(CFFarrays[1]):
        plt.hist(dataset, bins=20, color='blue', alpha=0.7)
    plt.title('ReE')

    # Plot ReHt histogram
    plt.subplot(2, 2, 3)
    for i, dataset in enumerate(CFFarrays[2]):
        plt.hist(dataset, bins=20, color='blue', alpha=0.7)
    plt.title('ReHt')

    # Plot dvcs histogram
    plt.subplot(2, 2, 4)
    for i, dataset in enumerate(CFFarrays[3]):
        plt.hist(dataset, bins=20, color='blue', alpha=0.7)
    plt.title('dvcs')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plot as a .pdf file
    plt.savefig(output_filename)

    # Show the plot
    plt.show()

# Call the function to plot histograms and save as .pdf
plot_histograms(CFFarrays, output_filename="histograms.pdf")
