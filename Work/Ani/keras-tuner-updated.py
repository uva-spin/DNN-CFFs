import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import BayesianOptimization
from sklearn.model_selection import train_test_split
import os

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    min_delta=0.001,
    mode='min',
    restore_best_weights=True
)

df = pd.read_csv('PseudoData_from_the_Basic_Model_for_JLab_Kinematics.csv')
X = df[['k', 'QQ', 'x_b', 't', 'phi_x']].values
CFFs = df[['ReH', 'ReE', 'ReHt', 'dvcs']].values
X_train, X_test, CFFs_train, CFFs_test = train_test_split(X, CFFs, test_size=0.2, random_state=42)

evaluation_df = df.iloc[1:25]
X_eval = evaluation_df[['k', 'QQ', 'x_b', 't', 'phi_x']].values
CFFs_eval = evaluation_df[['ReH', 'ReE', 'ReHt', 'dvcs']].values

def model_builder(hp):
    inputs = tf.keras.Input(shape=(5,), name='input_layer')
    x = inputs
    for i in range(hp.Int('num_layers', 1, 5)):
        x = tf.keras.layers.Dense(
            units=hp.Int('units_' + str(i), min_value=50, max_value=200, step=50),
            activation='tanh'
        )(x)
    outputs = tf.keras.layers.Dense(4, activation="linear", name='cff_output_layer')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.MeanSquaredError()
    )
    return model

tuner = BayesianOptimization(
    model_builder,
    objective='val_loss',
    max_trials=10,
    directory='keras_tuner_dir',
    project_name='keras_tuner_demo'
)

tuner.search(X_train, CFFs_train, epochs=100, validation_split=0.1, callbacks=[early_stopping])

def load_intermediate_model(model):
    return tf.keras.Model(inputs=model.input, outputs=model.get_layer('cff_output_layer').output)

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

best_models = tuner.get_best_models(num_models=10)
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
