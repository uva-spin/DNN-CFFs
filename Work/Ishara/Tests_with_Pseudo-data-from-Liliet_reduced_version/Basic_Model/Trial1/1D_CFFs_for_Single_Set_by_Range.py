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
grid_df = pd.read_csv('pseudo_basic_BKM10_Jlab_all_t2.csv')

# Filter the dataframe to include only set numbers from 1 to 5
grid_df = grid_df[grid_df['set'].isin(range(1, 6))]

# Initialize lists to store predictions for each model and set
all_cff_predictions = [[] for _ in range(len(models))]
all_f_predictions = [[] for _ in range(len(models))]

# Predict CFFs and F values for each model and set
for i, model_id in enumerate(models):
    tempFLayer, tempCFFsLayer = load_FLayer_and_cffLayer(model_id)
    for set_number in range(1, 6):
        prediction_inputs = grid_df[grid_df['set'] == set_number][['QQ', 'x_b', 't', 'phi_x', 'k']].head(1).to_numpy()
        cffs, f_values = predict_cffs_and_f(tempCFFsLayer, tempFLayer, prediction_inputs)
        all_cff_predictions[i].append(cffs)
        all_f_predictions[i].append(f_values)

# Create plots for each set
for set_number in range(1, 20):
    plt.figure(figsize=(15, 10))
    cff_labels = ['ReH', 'ReE', 'ReHt', 'dvcs']
    for i, cff_label in enumerate(cff_labels):
        plt.subplot(2, 2, i+1)
        data = np.array(all_cff_predictions)[:, set_number-1, :, i].flatten()
        plt.hist(data, bins=20, edgecolor='black', alpha=0.7, color='lightblue')
        
        mean_value = np.mean(data)
        std_deviation = np.std(data)
        
        # Plot vertical lines for true value, mean, and bounds for 1-sigma
        true_values = grid_df[grid_df['set'] == set_number][['ReH', 'ReE', 'ReHt', 'dvcs']].iloc[0].values
        plt.axvline(x=true_values[i], color='red', linestyle='--', label='True Value')
        plt.axvline(x=mean_value, color='blue', linestyle='--', label='Mean')
        plt.axvline(x=mean_value - std_deviation, color='green', linestyle='--', label='1-sigma')
        plt.axvline(x=mean_value + std_deviation, color='green', linestyle='--')
        
        # Fit a Gaussian curve with the correct x-axis limits
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mean_value, std_deviation)
        plt.plot(x, p * len(data) * (xmax - xmin) / 20, 'k', linewidth=2)
        
        plt.title(f'Set {set_number}: {cff_label} Histogram (from LMI)\nMean: {mean_value:.4f}, Std Dev: {std_deviation:.4f}')
        plt.xlabel(cff_label)
        plt.ylabel('Frequency')
        plt.legend()

    # Save the figure as a PDF file
    output_pdf_path_combined = f'CFFs_CombinedPlots_set{set_number}_subplot.pdf'
    plt.tight_layout()
    plt.savefig(output_pdf_path_combined)
    #plt.show()

    print(f"PDF file saved at: {output_pdf_path_combined}")
