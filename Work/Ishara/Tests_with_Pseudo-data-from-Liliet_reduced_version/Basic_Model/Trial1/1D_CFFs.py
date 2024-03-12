###################################
##  Written by Ishara Fernando   ##
##  Revised Date: 01/25/2024     ##
###################################

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BHDVCS_tf_modified import *

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
set_number = 1  # You can specify the desired set number
grid_df = pd.read_csv('pseudo_basic_BKM10_Jlab_all_t2.csv')
prediction_inputs = grid_df[grid_df['set'] == set_number][['QQ', 'x_b', 't', 'phi_x', 'k']].to_numpy()
set_data = grid_df[grid_df['set'] == set_number][['ReH', 'ReE', 'ReHt', 'dvcs']]

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
    plt.hist(np.array(cff_predictions)[:, :, i].T.flatten(), bins=20, edgecolor='black', alpha=0.7, color='lightblue')
    plt.axvline(x=true_values[i], color='red', linestyle='--', label='True Value')  # Vertical line for true value
    plt.title(f'{set_number}: {cff_label} Histogram with True Value')
    plt.xlabel(cff_label)
    plt.ylabel('Frequency')
    plt.legend()

# Save the figure as a PDF file
output_pdf_path_combined = 'CFFs_CombinedPlots_subplot.pdf'
plt.tight_layout()
plt.savefig(output_pdf_path_combined)
plt.show()

print(f"PDF file saved at: {output_pdf_path_combined}")




