############################################################################
###############  Written by Ishara Fernando                  ###############
##############  Revised Date: 10/14/2024    ################################
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
tf.keras.utils.get_custom_objects().update({'tanhshrink': tanhshrink})
import os
import sys
from scipy.stats import norm


def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")
        


data_file = 'Basic_Model_pseudo_data_for_Jlab_kinematics_with_sampling.csv'
#df = pd.read_csv(data_file, dtype=np.float64)
df = pd.read_csv(data_file)
df = df.rename(columns={"sigmaF": "errF"})

scratch_path = '/scratch/cee9hc/DNN_CFFs/Tests/test/'


    
# ########## Evaluation ######
# Load models
models_folder = str(scratch_path)+'DNNmodels_Kin_Set_'+str(j)
models = [os.path.join(models_folder, f) for f in os.listdir(models_folder) if f.endswith('.h5')]

# Take only one (or the first) line from grid_df
prediction_inputs = kin_df[kin_df['set'] == set_number].head(1)[['QQ', 'x_b', 't', 'phi_x', 'k']].to_numpy()
set_data = kin_df[kin_df['set'] == set_number].head(1)[['ReH', 'ReE', 'ReHt', 'dvcs']]

#print(len(models))

# Get true values
true_values = set_data.iloc[0].values
f_predictions = []
cff_predictions = []

# Predict CFFs and F values for each model
for model_id in models:
    tempFLayer, tempCFFsLayer = load_FLayer_and_cffLayer(model_id)
    cffs, f_values = predict_cffs_and_f(tempCFFsLayer, tempFLayer, prediction_inputs)
    f_predictions.append(f_values)
    cff_predictions.append(cffs)

# Create subplots in a single figure for CFF histograms with vertical lines
plt.figure(figsize=(15, 10))
cff_labels = ['ReH', 'ReE', 'ReHt', 'dvcs']

cffs_true_array = []
cffs_pred_array = []
cffs_stds_array = []

for i, cff_label in enumerate(cff_labels):
    plt.subplot(2, 2, i+1)
    data = np.array(cff_predictions)[:, :, i].T.flatten()
    plt.hist(data, bins=20, edgecolor='black', alpha=0.7, color='lightblue')

    mean_value = np.mean(data)
    std_deviation = np.std(data)
    
    cffs_true_array.append(true_values[i])
    cffs_pred_array.append(mean_value)
    cffs_stds_array.append(std_deviation)

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

    plt.title(f'Set {set_number}: {cff_label} Histogram (from Local)\n Mean: {mean_value:.4f}, Std Dev: {std_deviation:.4f}')
    plt.xlabel(cff_label)
    plt.ylabel('Frequency')
    plt.legend()

# Save the figure as a PDF file
output_pdf_path_combined = 'Comparison_Plots/'+'CFFs_CombinedPlots_subplot_kinematic_set_'+ str(j) +'.pdf'
plt.tight_layout()
plt.savefig(output_pdf_path_combined)
plt.close()


def EvalResults(j,cffs_true,cffs_pred,cffs_std):
    pseudodata_df = {'set': [],
                     'ReH_true': [],
                     'ReH_pred': [],
                     'ReH_res': [],
                     'ReH_std': [],
                     'ReE_true': [],
                     'ReE_pred': [],
                     'ReE_res': [],
                     'ReE_std': [],
                     'ReHt_true': [],
                     'ReHt_pred': [],
                     'ReHt_res': [],
                     'ReHt_std': [],
                     'dvcs_true': [],
                     'dvcs_pred': [],
                     'dvcs_res': [],
                     'dvcs_std': []}
    pseudodata_df['set'] = str(j)
    print(cffs_true)
    print(pseudodata_df['set'])
    pseudodata_df['ReH_true'] = cffs_true[0]
    pseudodata_df['ReH_pred'] = cffs_pred[0]
    pseudodata_df['ReH_res'] = np.abs(cffs_true[0]-cffs_pred[0])
    print(pseudodata_df['ReH_res'])
    pseudodata_df['ReH_std'] = cffs_std[0]
    pseudodata_df['ReE_true'] = cffs_true[1]
    pseudodata_df['ReE_pred'] = cffs_pred[1]
    pseudodata_df['ReE_res'] = np.abs(cffs_true[1]-cffs_pred[1])
    pseudodata_df['ReE_std'] = cffs_std[1]
    pseudodata_df['ReHt_true'] = cffs_true[2]
    pseudodata_df['ReHt_pred'] = cffs_pred[2]
    pseudodata_df['ReHt_res'] = np.abs(cffs_true[2]-cffs_pred[2])
    pseudodata_df['ReHt_std'] = cffs_std[2]
    pseudodata_df['dvcs_true'] = cffs_true[3]
    pseudodata_df['dvcs_pred'] = cffs_pred[3]
    pseudodata_df['dvcs_res'] = np.abs(cffs_true[3]-cffs_pred[3])
    pseudodata_df['dvcs_std'] = cffs_std[3]
    return pd.DataFrame(pseudodata_df)


#cffs_true_array = np.array(cffs_true_array)
#cffs_pred_array = np.array(cffs_pred_array)
#cffs_stds_array = np.array(cffs_stds_array)

#create_folders(str(scratch_path)+'CFFs_Evaluations')    
#tempresults=EvalResults(j,cffs_true_array,cffs_pred_array,cffs_stds_array)
#tempresults.to_csv(str(scratch_path)+'CFFs_Evaluations/'+'Eval_set_'+str(j)+'.csv', index=False)
#tempresults.to_csv('test.csv', index=False)