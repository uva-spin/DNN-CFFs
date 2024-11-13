############################################################################
#####  Written by Ishara Fernando, Ani Venkatapuram                  #######
##############  Revised Date: 10/23/2024    ################################
##### Rivanna usage: Run the following commands on your Rivanna terminal####
## source /home/lba9wf/miniconda3/etc/profile.d/conda.sh         ###########
## conda activate env                                            ###########
## pip3 install --user tensorflow-addons==0.21.0                 ###########
############################################################################
############################################################################
### This code is only performing the evaluation              ###############
### make sure to provide the correct path to the models folder #############
############################################################################

import numpy as np
import pandas as pd
import tensorflow as tf
from BHDVCS_tf_modified import *
import matplotlib.pyplot as plt
from tensorflow_addons.activations import tanhshrink
from scipy.stats import norm
import os
import sys

# Update custom object for TensorFlow
tf.keras.utils.get_custom_objects().update({'tanhshrink': tanhshrink})

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")

def load_FLayer_and_cffLayer(model):
    LayerF = tf.keras.models.load_model(model, custom_objects={'TotalFLayer': TotalFLayer})
    LayerCFFs = tf.keras.Model(inputs=LayerF.input, outputs=LayerF.get_layer('cff_output_layer').output)
    return LayerF, LayerCFFs

def predict_cffs_and_f(LayerCFFs, LayerF, inputs):
    cffs = LayerCFFs.predict(inputs)
    f_values = LayerF.predict(inputs)
    return cffs, f_values

# Load data
data_file = 'Basic_Model_pseudo_data_for_Jlab_kinematics_with_sampling.csv'
df = pd.read_csv(data_file)
df = df.rename(columns={"sigmaF": "errF"})

# Define the scratch path where the results will be saved
scratch_path = '/scratch/<your_uva_id>/DNN_CFFs/LocalFit_Tests/Test_01/'

create_folders('Comparison_Plots')
create_folders('CFF_Mean_Deviation_Plots')

# Variable to decide whether to use specific kinematic sets or all available sets
use_specific_sets = True  # Set to False if you want to search all available kinematic sets

# Define the list of specific kinematic sets you want to search for
specific_kin_sets = [1]

# If use_specific_sets is True, search for only the specific kinematic sets
if use_specific_sets:
    available_kin_sets = []
    for kin_set in specific_kin_sets:
        folder_name = f'DNNmodels_Kin_Set_{kin_set}'
        if folder_name in os.listdir(scratch_path):
            available_kin_sets.append(kin_set)

# If use_specific_sets is False, search for all available kinematic sets in the folder
else:
    available_kin_sets = [d for d in os.listdir(scratch_path) if d.startswith('DNNmodels_Kin_Set_')]
    available_kin_sets = [int(d.split('_')[-1]) for d in available_kin_sets]

available_kin_sets = sorted(available_kin_sets)
print(f"Available kinematic sets: {available_kin_sets}")

# Initialize an empty DataFrame to store all results
all_results_df = pd.DataFrame()

# **Step 1: Initialize a dictionary to store residuals for each CFF per kinematic set for violin plots**
#residuals_dict = {cff: {} for cff in ['ReH', 'ReE', 'ReHt', 'dvcs']}  # Each CFF will have residuals for each kinematic set

# ########## Loop over each kinematic set ##########
for j in available_kin_sets:
    print(f"Processing Kinematic Set: {j}")
    
    # Load models for the current kinematic set
    models_folder = str(scratch_path) + 'DNNmodels_Kin_Set_' + str(j)
    models = [os.path.join(models_folder, f) for f in os.listdir(models_folder) if f.endswith('.h5')]
    
    # Get the data for the current kinematic set
    kin_df = df[df['set'] == j]
    kin_df = kin_df.reset_index(drop=True)
    
    # Take the input values for the prediction (use first line of kin_df)
    prediction_inputs = kin_df[['QQ', 'x_b', 't', 'phi_x', 'k']].to_numpy()

    # Get the real F values and phi_x for the current kinematic set
    real_F_values = kin_df['F'].values
    phi_x_values = kin_df['phi_x'].values  # Extract the phi_x values from the dataframe

    # Get the true CFF values (assuming they are the same for all rows in the set)
    true_values = kin_df[['ReH', 'ReE', 'ReHt', 'dvcs']].iloc[0].values

    # List to store predictions from each replica for the current kinematic set
    f_predictions = []
    cff_predictions = []

    # Predict CFFs and F values for each model (replica) in the current kinematic set
    for model_id in models:
        tempFLayer, tempCFFsLayer = load_FLayer_and_cffLayer(model_id)
        cffs, f_values = predict_cffs_and_f(tempCFFsLayer, tempFLayer, prediction_inputs)
        f_predictions.append(f_values)
        cff_predictions.append(cffs)

    # ---- Chi-Square Calculation and F vs phi_x Plot ---- #
    # Initialize lists to store the mean and standard deviation for each phi_x
    mean_f_predictions = []
    std_f_predictions = []

    # Loop over each phi_x value
    for i in range(len(phi_x_values)):
        # Extract the predicted F values for the i-th phi_x from all replicas
        f_values_at_phi_x = [f_pred[i] for f_pred in f_predictions]
        mean_f_predictions.append(np.mean(f_values_at_phi_x))
        std_f_predictions.append(np.std(f_values_at_phi_x))

    mean_f_predictions = np.array(mean_f_predictions)
    std_f_predictions = np.array(std_f_predictions)

    # Calculate chi-square error
    chi_square_error = np.sum(((real_F_values - mean_f_predictions) / std_f_predictions) ** 2)

    # Save chi-square error to file
    chi_square_file = 'Comparison_Plots/chi_square_errors.txt'
    # Check if the file exists, if not create it and add the header
    if not os.path.exists(chi_square_file):
        with open(chi_square_file, 'w') as file:
            file.write("Kinematic Set\tChi-Square Error\n")  # Write header if the file doesn't exist
    # Append chi-square error data to the file
    with open(chi_square_file, 'a') as file:
        file.write(f"{j}\t{chi_square_error:.4f}\n")
    print(f"Kinematic Set {j}: Chi-Square Error = {chi_square_error:.4f}")

    # Create F vs phi_x plot
    plt.figure(figsize=(10, 6))
    plt.scatter(phi_x_values, real_F_values, color='red', label='Real F', zorder=5)
    plt.errorbar(phi_x_values, real_F_values, yerr=std_f_predictions, fmt='o', color='red', ecolor='red', capsize=5, label='Real F Error', zorder=6)
    plt.plot(phi_x_values, mean_f_predictions, color='blue', label='Mean F Prediction')
    plt.fill_between(phi_x_values, mean_f_predictions - std_f_predictions, mean_f_predictions + std_f_predictions, color='blue', alpha=0.2, label='±1 Std Dev')
    plt.xlabel('phi_x')
    plt.ylabel('F')
    plt.title(f'F vs phi_x with Error Bars and Error Bands for Kinematic Set {j}')
    plt.grid(True)
    plt.legend()

    # Save the F vs phi_x plot as a PNG file
    output_png_path = f'Comparison_Plots/F_vs_phi_x_with_error_bars_and_bands_Kinematic_Set_{j}.png'
    plt.savefig(output_png_path)
    plt.close()

    print(f"F vs phi_x plot saved: {output_png_path}")
    # ---- End of Chi-Square Calculation and Plot ---- #

    # Create subplots in a single figure for CFF histograms with vertical lines (if needed)
    plt.figure(figsize=(15, 10))
    cff_labels = ['ReH', 'ReE', 'ReHt', 'dvcs']

    # Initialize lists to store true, predicted, and std values
    cffs_true_array = []
    cffs_pred_array = []
    cffs_stds_array = []

    for i, cff_label in enumerate(cff_labels):
        plt.subplot(2, 2, i + 1)
        data = np.array(cff_predictions)[:, :, i].T.flatten()
        plt.hist(data, bins=20, edgecolor='black', alpha=0.7, color='lightblue')

        mean_value = np.mean(data)
        std_deviation = np.std(data)
        
        cffs_true_array.append(true_values[i])
        cffs_pred_array.append(mean_value)
        cffs_stds_array.append(std_deviation)

        # **Step 1.1: Store residuals for each CFF and kinematic set for violin plotting**
        # if j not in residuals_dict[cff_label]:
        #     residuals_dict[cff_label][j] = []
        # residuals_dict[cff_label][j].extend(np.abs(true_values[i] - data))  # Append residuals for this CFF and kinematic set

        # Plot vertical lines for true value, mean, and bounds for 1-sigma
        plt.axvline(x=true_values[i], color='red', linestyle='--', label='True Value')
        plt.axvline(x=mean_value, color='blue', linestyle='--', label='Mean')
        plt.axvline(x=mean_value - std_deviation, color='green', linestyle='--', label='1-sigma')
        plt.axvline(x=mean_value + std_deviation, color='green', linestyle='--')

        # Fit a Gaussian curve with the correct x-axis limits
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mean_value, std_deviation)
        plt.plot(x, p * len(data) * (xmax - xmin) / 20, 'k', linewidth=2)

        plt.title(f'Set {j}: {cff_label} Histogram (from Local)\n Mean: {mean_value:.4f}, Std Dev: {std_deviation:.4f}')
        plt.xlabel(cff_label)
        plt.ylabel('Frequency')
        plt.legend()

    # Save the figure as a PDF file
    output_pdf_path_combined = 'Comparison_Plots/' + 'CFFs_CombinedPlots_subplot_kinematic_set_' + str(j) + '.pdf'
    plt.tight_layout()
    plt.savefig(output_pdf_path_combined)
    plt.close()

    # Function to evaluate results and save to DataFrame
    def EvalResults(j, cffs_true, cffs_pred, cffs_std):
        pseudodata_df = {
            'set': [j],  
            'ReH_true': [cffs_true[0]],
            'ReH_pred': [np.mean(cffs_pred[0])],  
            'ReH_res': [np.abs(cffs_true[0] - np.mean(cffs_pred[0]))],  
            'ReH_std': [cffs_std[0]],
            'ReE_true': [cffs_true[1]],  
            'ReE_pred': [np.mean(cffs_pred[1])], 
            'ReE_res': [np.abs(cffs_true[1] - np.mean(cffs_pred[1]))],  
            'ReE_std': [cffs_std[1]],
            'ReHt_true': [cffs_true[2]],  
            'ReHt_pred': [np.mean(cffs_pred[2])],  
            'ReHt_res': [np.abs(cffs_true[2] - np.mean(cffs_pred[2]))],
            'ReHt_std': [cffs_std[2]],
            'dvcs_true': [cffs_true[3]],  
            'dvcs_pred': [np.mean(cffs_pred[3])],  
            'dvcs_res': [np.abs(cffs_true[3] - np.mean(cffs_pred[3]))],  
            'dvcs_std': [cffs_std[3]],
        }
        return pd.DataFrame(pseudodata_df)

    cffs_true_array = np.array(cffs_true_array)
    cffs_pred_array = np.array(cffs_pred_array)
    cffs_stds_array = np.array(cffs_stds_array)

    # Save evaluation results to CSV for the current kinematic set
    create_folders(str(scratch_path) + 'CFFs_Evaluations')    
    tempresults = EvalResults(j, cffs_true_array, cffs_pred_array, cffs_stds_array)
    tempresults.to_csv(str(scratch_path) + 'CFFs_Evaluations/' + 'Eval_set_' + str(j) + '.csv', index=False)

    # Append current results to the global DataFrame
    all_results_df = pd.concat([all_results_df, tempresults], ignore_index=True)

# Save the final DataFrame containing all results across all kinematic sets
all_results_df.to_csv('CFFs_AllSets_Combined.csv', index=False)

#Generating the plot for the kinematic sets 
def generate_mean_std_plots(df, cff_labels, output_dir='CFF_Mean_Deviation_Plots', sets_per_plot=20):
    """
    Generates and saves plots for each CFF showing the residual (true - predicted) and standard deviation.
    Each plot will contain up to a specified number of kinematic sets, and new plots will be generated 
    if the number of kinematic sets exceeds the limit.

    Parameters:
    - df: DataFrame containing the data for CFFs.
    - cff_labels: List of CFF labels (e.g., ['ReH', 'ReE', 'ReHt', 'dvcs']).
    - output_dir: Directory to save the plots.
    - sets_per_plot: Number of kinematic sets per plot.
    """

    for cff in cff_labels:
        res_col = f'{cff}_res'
        std_col = f'{cff}_std'
        sets = df['set']
        residuals = df[res_col]
        stds = df[std_col]

        # Break the sets into chunks to handle the sets_per_plot limit
        for chunk_start in range(0, len(sets), sets_per_plot):
            chunk_end = min(chunk_start + sets_per_plot, len(sets))
            chunk_sets = sets[chunk_start:chunk_end]
            chunk_residuals = residuals[chunk_start:chunk_end]
            chunk_stds = stds[chunk_start:chunk_end]

            plt.figure(figsize=(10, 6))

            for i, (set_num, res, std) in enumerate(zip(chunk_sets, chunk_residuals, chunk_stds)):
                # Plot the mean and standard deviation for each kinematic set in the chunk
                plt.plot([i + 1, i + 1], [res - std, res + std], color='blue', linewidth=2)  # Std bounds
                plt.scatter(i + 1, res, color='blue', zorder=5)  # Mean as a dot

            # Set plot labels and title
            plt.title(f'{cff} Residual with Standard Deviation (Sets {chunk_start + 1}-{chunk_end})')
            plt.xlabel('Kinematic Set')
            plt.ylabel(f'{cff} Residual')
            plt.xticks(range(1, chunk_end - chunk_start + 1), chunk_sets)  # Set x-tick labels to the kinematic set numbers
            plt.grid(True)

            # Save the plot for the current chunk of sets
            plot_path = os.path.join(output_dir, f'{cff}_Residual_Std_Plot_Sets_{chunk_start + 1}_to_{chunk_end}.png')
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

# Example usage (with 20 sets per plot):
generate_mean_std_plots(all_results_df, cff_labels=['ReH', 'ReE', 'ReHt', 'dvcs'], sets_per_plot=20)

df_kinematic = pd.read_csv("Basic_Model_pseudo_data_for_Jlab_kinematics_with_sampling.csv")

df_cffs = pd.read_csv("CFFs_AllSets_Combined.csv")

df_evaluation = pd.DataFrame()

for set_id in df_cffs['set'].unique():
    df_kin_set = df_kinematic[df_kinematic['set'] == set_id].copy()
    df_cffs_set = df_cffs[df_cffs['set'] == set_id].copy()
    df_f_phi = pd.read_csv(f"Comparison_Plots/F_vs_phi_x_Kinematic_Set_{set_id}.csv")
    
    df_cffs_repeated = pd.concat([df_cffs_set] * 24, ignore_index=True)
    
    df_kin_set.reset_index(drop=True, inplace=True)
    df_f_phi.reset_index(drop=True, inplace=True)
    df_cffs_repeated.reset_index(drop=True, inplace=True)
    
    df_combined = pd.concat([df_kin_set, df_f_phi[['Mean F Prediction', 'Std Dev Prediction']], df_cffs_repeated], axis=1)
    
    df_evaluation = pd.concat([df_evaluation, df_combined], ignore_index=True)

df_evaluation.to_csv("evaluation.csv", index=False)





df_evaluation = pd.read_csv("evaluation.csv")

df_evaluation['set'] = df_evaluation['set'].astype(int)

columns_to_drop = ['ReH_true', 'ReE_true', 'ReHt_true', 'dvcs_true']
df_evaluation = df_evaluation.drop(columns=columns_to_drop)

ordered_columns = [
    'set', 'k', 'QQ', 'x_b', 't', 'phi_x', 'F', 'sigmaF', 
    'Mean F Prediction', 'Std Dev Prediction', 
    'ReH', 'ReH_pred', 'ReH_res', 'ReH_std',
    'ReE', 'ReE_pred', 'ReE_res', 'ReE_std',
    'ReHt', 'ReHt_pred', 'ReHt_res', 'ReHt_std',
    'dvcs', 'dvcs_pred', 'dvcs_res', 'dvcs_std'
]

df_evaluation = df_evaluation[ordered_columns]

df_evaluation.to_csv("evaluation.csv", index=False)


# **Step 2: Generate Violin Plots based on residuals**
# Function to generate and save violin plots for each CFF using real residuals
# def generate_violin_plots_for_cffs(residuals_dict, cff_labels, output_dir='ViolinPlots', max_sets_per_plot=10):
#     """
#     Generates and saves violin plots for each CFF using the real residuals from all kinematic sets.
    
#     Parameters:
#     - residuals_dict: Dictionary containing residuals for each CFF label.
#     - cff_labels: List of CFF labels (e.g., ['ReH', 'ReE', 'ReHt', 'dvcs']).
#     - output_dir: Directory to save the violin plots.
#     - max_sets_per_plot: Maximum number of kinematic sets to display per plot.
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     for cff_label in cff_labels:
#         # Get the available kinematic sets
#         available_sets = sorted(residuals_dict[cff_label].keys())
#         total_sets = len(available_sets)
        
#         # Split the kinematic sets into chunks of size max_sets_per_plot
#         for plot_index, chunk_start in enumerate(range(0, total_sets, max_sets_per_plot)):
#             chunk_end = min(chunk_start + max_sets_per_plot, total_sets)
#             kinematic_chunk = available_sets[chunk_start:chunk_end]

#             # Extract residuals for the current chunk of kinematic sets for the current CFF
#             residuals = [residuals_dict[cff_label][kin_set] for kin_set in kinematic_chunk]

#             # Create the violin plot
#             plt.figure(figsize=(10, 6))
#             parts = plt.violinplot(residuals, widths=0.7, showmeans=True, showmedians=False, bw_method=0.5)

#             for pc in parts['bodies']:
#                 pc.set_facecolor('gray')
#                 pc.set_edgecolor('black')
#                 pc.set_alpha(0.7)

#             plt.axhline(y=0, color='black', linestyle='--', linewidth=1)  # Residuals centered around 0
#             plt.xlabel('Kinematic Sets')
#             plt.ylabel(f'{cff_label} Residual')
#             plt.title(f'Violin Plot of {cff_label} Residuals (Sets {chunk_start + 1} to {chunk_end})')

#             # Set x-ticks to correspond to the kinematic sets in the current chunk
#             plt.xticks(ticks=range(1, len(kinematic_chunk) + 1), labels=[str(kin_set) for kin_set in kinematic_chunk])

#             # Save the plot
#             plot_filename = f'{cff_label}_Residuals_ViolinPlot_{plot_index + 1}.png'
#             plot_path = os.path.join(output_dir, plot_filename)
#             plt.tight_layout()
#             plt.savefig(plot_path)
#             plt.close()
#             print(f"Violin plot saved: {plot_path}")

# # **Step 3: Call the function to generate violin plots**
# generate_violin_plots_for_cffs(residuals_dict, ['ReH', 'ReE', 'ReHt', 'dvcs'])
