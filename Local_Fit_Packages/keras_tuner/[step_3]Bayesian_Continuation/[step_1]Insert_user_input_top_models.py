import keras_tuner as kt
import tensorflow as tf
import re
import os
from BHDVCS_tf_modified import TotalFLayer
from user_inputs import *

# Define the model builder exactly as used during tuning
def build_model(hp):
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
    inputs = tf.keras.Input(shape=(5,), name='input_layer')
    QQ, x_b, t, phi, k = tf.split(inputs, num_or_size_splits=5, axis=1)
    kinematics = tf.keras.layers.concatenate([QQ, x_b, t])

    regularizer_choice = hp.Choice("regularizer", ["l1", "l2", "none"])
    if regularizer_choice == "l1":
        regularizer = tf.keras.regularizers.l1(1e-5)
    elif regularizer_choice == "l2":
        regularizer = tf.keras.regularizers.l2(1e-5)
    else:
        regularizer = None

    x = kinematics
    for i in range(hp.Int("num_layers", 4, 10)):
        x = tf.keras.layers.Dense(
            units=hp.Int(f"units_{i}", min_value=64, max_value=1024, step=64),
            activation=hp.Choice("activation", ["relu", "relu6", "tanh", "selu", "elu"]),
            kernel_initializer=initializer,
            kernel_regularizer=regularizer
        )(x)

    outputs = tf.keras.layers.Dense(4, activation="linear", kernel_initializer=initializer)(x)
    total_inputs = tf.keras.layers.concatenate([inputs, outputs], axis=1)
    TotalF = TotalFLayer(name='TotalFLayer')(total_inputs)

    model = tf.keras.Model(inputs=inputs, outputs=TotalF)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice("learning_rate", [1e-3, 5e-4, 1e-4, 5e-5, 1e-5])
        ),
        loss="mse"
    )
    return model

# Load the Bayesian tuner
tuner = kt.BayesianOptimization(
    hypermodel=build_model,
    objective="val_loss",
    max_trials=50,
    directory="bayesian_tuning",
    project_name="dnn_optimization"
)
tuner.reload()

# Extract top 10 trial configurations
best_trials = tuner.oracle.get_best_trials(num_trials=10)
top_hparams_dicts = [trial.hyperparameters.values for trial in best_trials]

# Format to Python syntax
formatted_output = "bayesian_model_hparams = [\n"
for entry in top_hparams_dicts:
    formatted_output += f"    {entry},\n"
formatted_output += "]\n"

# Update or insert into user_inputs.py
user_inputs_path = "user_inputs.py"
if os.path.exists(user_inputs_path):
    with open(user_inputs_path, "r") as f:
        content = f.read()
else:
    content = ""

if "bayesian_model_hparams" in content:
    content = re.sub(
        r"bayesian_model_hparams\s*=\s*\[.*?\]\n",
        formatted_output,
        content,
        flags=re.DOTALL
    )
else:
    content += "\n\n" + formatted_output

with open(user_inputs_path, "w") as f:
    f.write(content)

print("Inserted top 10 Bayesian model hyperparameters into user_inputs.py as 'bayesian_model_hparams'.")
