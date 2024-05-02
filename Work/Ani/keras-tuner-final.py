import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner
from sklearn.model_selection import train_test_split
import os

# Load data
df = pd.read_csv('PseudoData_from_the_Basic_Model_for_JLab_Kinematics.csv')
X = df[['k', 'QQ', 'x_b', 't', 'phi_x']].values
CFFs = df[['ReH', 'ReE', 'ReHt', 'dvcs']].values
F = df['F'].values  # Assuming 'F' is a column in your DataFrame

# Split data for training and validation for the entire dataset
X_train, X_test, CFFs_train, CFFs_test, F_train, F_test = train_test_split(X, CFFs, F, test_size=0.2, random_state=42)

# Specific subset for CFF detailed evaluation
evaluation_df = df.iloc[1:25]  # Subset for evaluation
X_eval = evaluation_df[['k', 'QQ', 'x_b', 't', 'phi_x']].values
CFFs_eval = evaluation_df[['ReH', 'ReE', 'ReHt', 'dvcs']].values

# Define the combined metric
def combined_metric(y_true, y_pred):
    F_true, CFFs_true = y_true[0], y_true[1]
    F_pred, CFFs_pred = y_pred[0], y_pred[1]

    mse_F = tf.keras.losses.mean_squared_error(F_true, F_pred)
    # Normalize MSE assuming max possible MSE (based on range of F) is known, adjust `max_mse_F` as needed
    max_mse_F = tf.reduce_max(F_true)**2  
    normalized_mse_F = 1 - (mse_F / max_mse_F)

    safe_denominator = tf.clip_by_value(CFFs_true, tf.keras.backend.epsilon(), tf.reduce_max(CFFs_true))
    percent_errors = tf.abs((CFFs_pred - CFFs_true) / safe_denominator)
    # Mean percent error across all CFFs
    mean_pe = tf.reduce_mean(percent_errors)
    pe_score = 1 / (1 + mean_pe)  

    # Compute standard deviation for each CFF
    std_devs = tf.math.reduce_std(CFFs_pred - CFFs_true)
    sd_score = 1 / (1 + std_devs)  
    #Take this sd_score out because it doesn't make sense for the statistical uncertainty

    # Combine scores, balancing the MSE of F and the custom score for CFFs
    return 0.5 * normalized_mse_F + 0.5 * pe_score# + 0.25 * sd_score #take out sd_score and recalculate to add up to 1

# Early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    min_delta=0.001,
    mode='min',
    restore_best_weights=True
)

# Model builder
def model_builder(hp):
    inputs = keras.Input(shape=(5,), name='input_layer')
    x = inputs
    for i in range(hp.Int('num_layers', 1, 5)):
        x = layers.Dense(units=hp.Int('units_' + str(i), min_value=50, max_value=200, step=50), activation='tanh')(x)
    cff_outputs = layers.Dense(4, activation="linear", name='cff_output_layer')(x)
    f_output = layers.Dense(1, activation="linear", name='total_F_layer')(inputs)  # Directly from inputs for simplicity
    model = keras.Model(inputs=inputs, outputs=[f_output, cff_outputs])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss={'total_F_layer': 'mean_squared_error', 'cff_output_layer': 'mean_squared_error'},
        metrics={'total_F_layer': 'mean_squared_error', 'cff_output_layer': combined_metric}
    )
    return model

# Setup the tuner
tuner = keras_tuner.BayesianOptimization(
    model_builder,
    objective=keras_tuner.Objective("val_total_F_layer_mean_squared_error", direction="min"),
    max_trials=10,
    executions_per_trial=1,
    directory='keras_tuner_dir',
    project_name='keras_tuner_demo'
)

# Search for the optimal hyperparameters
tuner.search(X_train, {'total_F_layer': F_train, 'cff_output_layer': CFFs_train},
             validation_data=(X_test, {'total_F_layer': F_test, 'cff_output_layer': CFFs_test}),
             epochs=100, callbacks=[early_stopping])

# Load intermediate model for CFF output
def load_intermediate_model(model):
    return keras.Model(inputs=model.input, outputs=model.get_layer('cff_output_layer').output)

# Custom evaluation focused on the CFF subset
def custom_evaluation(full_model, x_test, y_test, model_id):
    intermediate_model = load_intermediate_model(full_model)
    cff_predictions = intermediate_model.predict(x_test)
    results = {}
    good_cff_count = 0
    for i, cff_name in enumerate(['ReH', 'ReE', 'ReHt', 'dvcs']):
        mean_predicted = np.mean(cff_predictions[:, i])
        mean_actual = np.mean(y_test[:, i])
        std_dev = np.std(cff_predictions[:, i])
        percent_diff = np.abs((mean_predicted - mean_actual) / mean_actual) * 100
        results[cff_name] = {
            'Mean Predicted': mean_predicted,
            'Mean Actual': mean_actual,
            'Standard Deviation': std_dev,
            'Percent Difference': percent_diff
        }
        if percent_diff < 10 and std_dev < 0.75:
            good_cff_count += 1
    if good_cff_count >= 2:
        folder_name = "2_good_CFFs" if good_cff_count == 2 else "3_good_CFFs"
        save_path = os.path.join(folder_name, f'model_{model_id}.h5')
        os.makedirs(folder_name, exist_ok=True)
        full_model.save(save_path)
        print(f"Model {model_id} saved to {save_path} with {good_cff_count} good CFF predictions")
    return {
        'model_id': model_id,
        'results': results,
        'good_cff_count': good_cff_count
    }

# Evaluate and store results
best_models = tuner.get_best_models(num_models=5)
evaluation_results = [custom_evaluation(model, X_eval, CFFs_eval, i) for i, model in enumerate(best_models)]
good_models = [result for result in evaluation_results if result['good_cff_count'] >= 2]

if not good_models:
    sorted_models = sorted(evaluation_results, key=lambda x: np.mean([v['Percent Difference'] for v in x['results'].values()]))
    best_three = sorted_models[:3]
    folder_name = "3_best"
    os.makedirs(folder_name, exist_ok=True)
    for result in best_three:
        model_id = result['model_id']
        model = best_models[model_id]
        save_path = os.path.join(folder_name, f'model_{model_id}.h5')
        model.save(save_path)
        print(f"Model {model_id} saved to {save_path} with statistics:")
        for cff_name, stats in result['results'].items():
            print(f"  {cff_name}: Mean Predicted = {stats['Mean Predicted']:.4f}, Mean Actual = {stats['Mean Actual']:.4f}, Std Dev = {stats['Standard Deviation']:.4f}, Percent Diff = {stats['Percent Difference']:.4f}")