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

model_path = '3_good_CFFs/model_4.h5'
model = tf.keras.models.load_model(model_path, custom_objects={'combined_metric': combined_metric})

df = pd.read_csv('PseudoData_from_the_Basic_Model_for_JLab_Kinematics.csv')
df = df.iloc[1:25]  # Subset for detailed evaluation

inputs = df[['k', 'QQ', 'x_b', 't', 'phi_x']].to_numpy()

# Predict using the model
predictions = model.predict(inputs)
f_predictions, cff_predictions = predictions[0], predictions[1]  # Assuming the first output is F and the second is CFFs

actual_means = {
    'ReE': df['ReE'].mean(),
    'ReH': df['ReH'].mean(),
    'ReHt': df['ReHt'].mean(),
    'dvcs': df['dvcs'].mean()
}

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

# Plot histograms for each CFF
cff_names = ['ReE', 'ReH', 'ReHt', 'dvcs']
for i, cff_name in enumerate(cff_names):
    plot_histograms(cff_predictions[:, i], actual_means, cff_name)
