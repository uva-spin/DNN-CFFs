import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

# Load Data
df = pd.read_csv('PseudoData_from_the_Basic_Model_for_JLab_Kinematics.csv')
X = df[['k', 'QQ', 'x_b', 't', 'phi_x']].values
CFFs = df[['ReH', 'ReE', 'ReHt', 'dvcs']].values
F = df['F'].values

# Split the data
X_train, X_test, CFFs_train, CFFs_test, F_train, F_test = train_test_split(X, CFFs, F, test_size=0.2, random_state=42)

# Define the model architecture
def build_model():
    input_layer = keras.Input(shape=(5,))
    x = keras.layers.Dense(100, activation='tanh')(input_layer)
    x = keras.layers.Dense(50, activation='tanh')(x)
    x = keras.layers.Dense(50, activation='tanh')(x)
    x = keras.layers.Dense(200, activation='tanh')(x)
    x = keras.layers.Dense(50, activation='tanh')(x)
    cff_output = keras.layers.Dense(4, activation='linear', name='cff_output_layer')(x)
    model = keras.Model(inputs=input_layer, outputs=cff_output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[combined_metric])
    return model

# Combined metric function
def combined_metric(y_true, y_pred):
    F_true, CFFs_true = y_true[0], y_true[1]
    F_pred, CFFs_pred = y_pred[0], y_pred[1]

    # MSE for the F predictions
    mse_F = tf.keras.losses.mean_squared_error(F_true, F_pred)
    normalized_mse_F = 1 - (mse_F / tf.reduce_max(mse_F))

    # Prevent division by zero in CFF calculations
    safe_CFFs_true = tf.clip_by_value(CFFs_true, tf.keras.backend.epsilon(), tf.reduce_max(CFFs_true))

    # Calculate percent errors for each CFF
    percent_errors = tf.abs((CFFs_pred - safe_CFFs_true) / safe_CFFs_true)
    mean_percent_error = tf.reduce_mean(percent_errors)

    # Creating a score that approaches 1 as the mean_percent_error approaches 0
    pe_score = 1 / (1 + mean_percent_error)

    # Combine the MSE score for F and the percent error score for CFFs
    return 0.5 * normalized_mse_F + 0.5 * pe_score

# Check if the directory exists, if not, create it
model_dir = 'model_replicas'
os.makedirs(model_dir, exist_ok=True)

# Training function
def train_model(i):
    print(f"Training model {i+1}")
    model = build_model()
    model.fit(X_train, CFFs_train, epochs=10, verbose=0)  # Reduce epochs for demo purposes
    # Save each model to the specified directory
    model_path = os.path.join(model_dir, f'model_{i+1}.h5')
    model.save(model_path)
    print(f"Model {i+1} saved at {model_path}.")

# Train replicas using multiprocessing
if __name__ == '__main__':
    num_replicas = 200
    with Pool(10) as p:  # Adjust the number of processes based on your CPU
        p.map(train_model, range(num_replicas))
