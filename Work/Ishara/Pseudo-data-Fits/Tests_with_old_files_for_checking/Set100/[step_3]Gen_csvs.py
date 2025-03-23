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
from user_inputs import *
from DNN_model import *
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import sys

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

def remove_extra_evaluation_columns(input_csv, output_csv):
    """
    Remove specific columns from a CSV file if they exist and rename columns if they exist.

    Parameters:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to the output CSV file with specified columns removed or renamed.
    """
    df = pd.read_csv(input_csv)

    columns_to_remove = ['Mean F Prediction', 'Std Dev Prediction', 'ReH_std', 'ReE_std', 'ReHt_std', 'dvcs_std']
    
    # Remove columns only if they exist
    existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]
    if existing_columns_to_remove:
        df.drop(columns=existing_columns_to_remove, inplace=True)

    # Define column renaming mapping
    rename_mapping = {
        "ReH_pred": "ReH",
        "ReE_pred": "ReE",
        "ReHt_pred": "ReHt",
        "dvcs_pred": "dvcs"
    }

    # Rename columns only if they exist
    existing_rename_mapping = {old: new for old, new in rename_mapping.items() if old in df.columns}
    if existing_rename_mapping:
        df.rename(columns=existing_rename_mapping, inplace=True)

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    
    print(f"Columns removed: {existing_columns_to_remove}")
    print(f"Columns renamed: {existing_rename_mapping}")
    print(f"File saved as {output_csv}")

remove_extra_evaluation_columns(initial_data_file, "temp_pseudodata.csv")



# Load data
data_file = 'temp_pseudodata.csv'
df = pd.read_csv(data_file)
df = df.rename(columns={"sigmaF": "errF"})


#create_folders('Comparison_Plots_Step3')
#create_folders('CFF_Mean_Deviation_Plots')

# Variable to decide whether to use specific kinematic sets or all available sets
use_specific_sets = False  # Set to False if you want to search all available kinematic sets

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

    output_csv_path = f'Comparison_Plots/F_vs_phi_x_Kinematic_Set_{j}.csv'
    f_vs_phi_data = {
        'phi_x': phi_x_values,
        'Real F': real_F_values,
        'Mean F Prediction': mean_f_predictions,
        'Std Dev Prediction': std_f_predictions
    }
    f_vs_phi_df = pd.DataFrame(f_vs_phi_data)
    f_vs_phi_df.to_csv(output_csv_path, index=False)

    print(f"F vs phi_x csvs saved: {output_csv_path}")
    # ---- End of Chi-Square Calculation and Plot ---- #

    # Create subplots in a single figure for CFF histograms with vertical lines (if needed)
    plt.figure(figsize=(15, 10))
    cff_labels = ['ReH', 'ReE', 'ReHt', 'dvcs']

    # Initialize lists to store true, predicted, and std values
    cffs_true_array = []
    cffs_pred_array = []
    cffs_stds_array = []
    csv_path = os.path.join(scratch_path, f'CFFs_Predictions_Set_{j}.csv')
    all_cff_predictions_df = pd.DataFrame()
    for i, cff_label in enumerate(cff_labels):
        plt.subplot(2, 2, i + 1)
        data = np.array(cff_predictions)[:, :, i].T.flatten()
        plt.hist(data, bins=20, edgecolor='black', alpha=0.7, color='lightblue')

        print(f"\nPredictions for {cff_label}:")
        print(data)
        print("\n" + "-" * 40)

        mean_value = np.mean(data)
        std_deviation = np.std(data)
        
        cffs_true_array.append(true_values[i])
        cffs_pred_array.append(mean_value)
        cffs_stds_array.append(std_deviation)

        # Prepare data for saving to CSV
        predictions_df = pd.DataFrame({
            'set': [j] * len(data),
            'cff_label': [cff_label] * len(data),
            'prediction': data,
            'true_value': [true_values[i]] * len(data),
            'mean_value': [mean_value] * len(data),
            'std_deviation': [std_deviation] * len(data)
        })

        # Append to the overall DataFrame
        all_cff_predictions_df = pd.concat([all_cff_predictions_df, predictions_df], ignore_index=True)
        all_cff_predictions_df.to_csv(csv_path, index=False)
        print(f"Predictions CSV saved: {csv_path}")

        # **Step 1.1: Store residuals for each CFF and kinematic set for violin plotting**
        # if j not in residuals_dict[cff_label]:
        #     residuals_dict[cff_label][j] = []
        # residuals_dict[cff_label][j].extend(np.abs(true_values[i] - data))  # Append residuals for this CFF and kinematic set

        

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
all_results_df.to_csv('Summary_of_CFFs.csv', index=False)


df_kinematic = pd.read_csv("temp_pseudodata.csv")

df_cffs = pd.read_csv("Summary_of_CFFs.csv")

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

df_evaluation.to_csv("results.csv", index=False)