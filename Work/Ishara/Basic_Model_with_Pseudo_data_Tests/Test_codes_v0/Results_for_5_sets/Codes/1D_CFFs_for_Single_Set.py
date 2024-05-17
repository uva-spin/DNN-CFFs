###################################
##  Written by Ishara Fernando   ##
##  Revised Date: 02/05/2024     ##
###################################

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BHDVCS_tf_modified import *
from scipy.stats import norm
from tensorflow_addons.activations import tanhshrink
tf.keras.utils.get_custom_objects().update({'tanhshrink': tanhshrink})

def load_FLayer_and_cffLayer(model):
    LayerF = tf.keras.models.load_model(model, custom_objects={'TotalFLayer': TotalFLayer})
    LayerCFFs = tf.keras.Model(inputs=LayerF.input, outputs=LayerF.get_layer('cff_output_layer').output)
    return LayerF, LayerCFFs

def predict_cffs_and_f(LayerCFFs, LayerF, inputs):
    cffs = LayerCFFs.predict(inputs)
    f_values = LayerF.predict(inputs)
    return cffs, f_values

# Load models
models_folder = 'DNNmodels'
models = [os.path.join(models_folder, f) for f in os.listdir(models_folder) if f.endswith('.h5')]

# Load prediction inputs from CSV
set_number = 5  # You can specify the desired set number
grid_df = pd.read_csv('Basic_Model_pseudo_data_for_Jlab_kinematics.csv')
grid_df = grid_df[grid_df['set'] == set_number]
grid_df = grid_df.reset_index(drop=True)

# Take only one (or the first) line from grid_df
prediction_inputs = grid_df[grid_df['set'] == set_number].head(1)[['QQ', 'x_b', 't', 'phi_x', 'k']].to_numpy()
set_data = grid_df[grid_df['set'] == set_number].head(1)[['ReH', 'ReE', 'ReHt', 'dvcs']]

# Get true values
true_values = set_data.iloc[0].values

# Initialize lists to store predictions
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
for i, cff_label in enumerate(cff_labels):
    plt.subplot(2, 2, i+1)
    data = np.array(cff_predictions)[:, :, i].T.flatten()
    plt.hist(data, bins=20, edgecolor='black', alpha=0.7, color='lightblue')
    
    mean_value = np.mean(data)
    std_deviation = np.std(data)
    
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
    
    plt.title(f'Set {set_number}: {cff_label} Histogram (from Local)\nMean: {mean_value:.4f}, Std Dev: {std_deviation:.4f}')
    plt.xlabel(cff_label)
    plt.ylabel('Frequency')
    plt.legend()

# Save the figure as a PDF file
output_pdf_path_combined = 'CFFs_CombinedPlots_subplot.pdf'
plt.tight_layout()
plt.savefig(output_pdf_path_combined)
#plt.show()

print(f"PDF file saved at: {output_pdf_path_combined}")

