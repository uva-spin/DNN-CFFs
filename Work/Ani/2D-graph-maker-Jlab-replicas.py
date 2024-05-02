import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def combined_metric(y_true, y_pred):
    F_true, CFFs_true = y_true[:, 0], y_true[:, 1:]
    F_pred, CFFs_pred = y_pred[:, 0], y_pred[:, 1:]
    mse_F = tf.keras.losses.mean_squared_error(F_true, F_pred)
    normalized_mse_F = 1 - (mse_F / tf.reduce_max(mse_F))
    safe_denominator = tf.clip_by_value(CFFs_true, tf.keras.backend.epsilon(), None)
    percent_errors = tf.abs((CFFs_pred - CFFs_true) / safe_denominator)
    pe_score = tf.reduce_mean(1 / (1 + tf.reduce_mean(percent_errors, axis=0)))
    return 0.5 * normalized_mse_F + 0.5 * pe_score

def load_model_custom_metric(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={'combined_metric': combined_metric})

# Load data
df = pd.read_csv('PseudoData_from_the_Basic_Model_for_JLab_Kinematics.csv')
df = df.iloc[1:25]  # Subset for detailed evaluation
inputs = df[['k', 'QQ', 'x_b', 't', 'phi_x']].to_numpy()

# Directory containing all models
model_dir = 'model_replicas'
model_files = [os.path.join(model_dir, file) for file in os.listdir(model_dir) if file.endswith('.h5')]

# Collect all predictions
all_cff_predictions = {cff_name: [] for cff_name in ['ReE', 'ReH', 'ReHt', 'dvcs']}

for model_path in model_files:
    model = load_model_custom_metric(model_path)
    predictions = model.predict(inputs)
    if len(predictions.shape) == 2 and predictions.shape[1] == 4:
        cff_predictions = predictions  # If predictions directly provide the 4 CFF outputs
    else:
        raise ValueError("Unexpected output shape from model predictions: {}".format(predictions.shape))
    
    for i, cff_name in enumerate(['ReE', 'ReH', 'ReHt', 'dvcs']):
        all_cff_predictions[cff_name].extend(cff_predictions[:, i])

# Plot histograms for each CFF
for cff_name in ['ReE', 'ReH', 'ReHt', 'dvcs']:
    plt.figure(figsize=(8, 6))
    plt.hist(all_cff_predictions[cff_name], bins=30, alpha=0.7, color='skyblue', edgecolor='black', label=f'Predicted {cff_name}')
    actual_mean = df[cff_name].mean()
    predicted_mean = np.mean(all_cff_predictions[cff_name])
    plt.axvline(actual_mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean Actual: {actual_mean:.2f}')
    plt.axvline(predicted_mean, color='green', linestyle='dashed', linewidth=1, label=f'Mean Predicted: {predicted_mean:.2f}')
    plt.title(f'{cff_name} Predictions Histogram Across Models')
    plt.xlabel(f'{cff_name} Values')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
