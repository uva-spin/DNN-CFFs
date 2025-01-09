import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from BHDVCS_tf_modified import TotalFLayer

def load_intermediate_model(model_path):
    custom_objects = {'TotalFLayer': TotalFLayer}
    full_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    intermediate_model = tf.keras.Model(inputs=full_model.input,
                                        outputs=full_model.get_layer('cff_output_layer').output)
    return intermediate_model

def plot_histograms(cff_predictions, actual_means, cff_name):
    plt.figure(figsize=(8, 6))
    plt.hist(cff_predictions, bins=30, alpha=0.7, color='skyblue', edgecolor='black', label=f'Predicted {cff_name}')
    plt.axvline(np.mean(cff_predictions), color='red', linestyle='dashed', linewidth=1, label=f'Mean Predicted: {np.mean(cff_predictions):.2f}')
    plt.axvline(actual_means[cff_name], color='green', linestyle='dashed', linewidth=1, label=f'Mean Actual: {actual_means[cff_name]:.2f}')
    plt.title(f'{cff_name} Predictions Histogram')
    plt.xlabel(f'{cff_name} Values')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Load the data and prepare inputs
df = pd.read_csv('PseudoData_from_the_Basic_Model_for_JLab_Kinematics.csv')
df = df.iloc[1:25]  # Slice the DataFrame to only use rows 2 to 25

inputs = df[['k', 'QQ', 'x_b', 't', 'phi_x']].to_numpy()

# Actual mean values of CFFs from the dataset
actual_means = {
    'ReE': df['ReE'].mean(),
    'ReH': df['ReH'].mean(),
    'ReHt': df['ReHt'].mean(),
    'dvcs': df['dvcs'].mean()
}

# Load the intermediate model
model_path = '3_good_CFFs/model_4.h5'
intermediate_model = load_intermediate_model(model_path)

# Predict CFFs using the intermediate model
cff_predictions = intermediate_model.predict(inputs)

# Plot histograms for each CFF
cff_names = ['ReE', 'ReH', 'ReHt', 'dvcs']
for i, cff_name in enumerate(cff_names):
    plot_histograms(cff_predictions[:, i], actual_means, cff_name)
