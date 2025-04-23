import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from BHDVCS_tf_modified import *
from user_inputs import *

# Enable Multi-GPU Training
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

def build_model(hp):
    with strategy.scope():
        initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
        inputs = tf.keras.Input(shape=(5), name='input_layer')
        QQ, x_b, t, phi, k = tf.split(inputs, num_or_size_splits=5, axis=1)
        kinematics = tf.keras.layers.concatenate([QQ, x_b, t])

        # Tune number of layers and units per layer
        x = kinematics
        for i in range(hp.Int("num_layers", 4, 6)):
            x = tf.keras.layers.Dense(
                units=hp.Int(f"units_{i}", min_value=64, max_value=512, step=64),
                activation=hp.Choice("activation", ["relu", "relu6", "tanh"]),
                kernel_initializer=initializer
            )(x)

        outputs = tf.keras.layers.Dense(
            4, activation="linear", kernel_initializer=initializer, name='cff_output_layer'
        )(x)
        total_FInputs = tf.keras.layers.concatenate([inputs, outputs], axis=1)
        TotalF = TotalFLayer(name='TotalFLayer')(total_FInputs)

        model = tf.keras.Model(inputs=inputs, outputs=TotalF, name="tfmodel")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice("learning_rate", [1e-3, 5e-4, 1e-4, 5e-5, 1e-5])
            ),
            loss=tf.keras.losses.MeanSquaredError()
        )
    return model

# Load dataset
df = pd.read_csv(initial_data_file)
df = df.rename(columns={"sigmaF": "errF"})
df = df[df["F"] != 0]

# Prepare training data
X = df[['QQ', 'x_b', 't', 'phi_x', 'k']]
y = df['F']
yerr = df['errF']

def split_data(X, y, yerr, split=0.1):
    temp = np.random.choice(list(range(len(y))), size=int(len(y) * split), replace=False)
    test_X = X.iloc[temp]
    train_X = X.drop(temp)
    test_y = y.iloc[temp]
    train_y = y.drop(temp)
    test_yerr = yerr.iloc[temp]
    train_yerr = yerr.drop(temp)
    return train_X, test_X, train_y, test_y, train_yerr, test_yerr

train_X, test_X, train_y, test_y, _, _ = split_data(X, y, yerr)

# Set up Keras Tuner
tuner = kt.Hyperband(
    build_model,
    objective="val_loss",
    max_epochs=3000,
    factor=4,
    directory="hyperband_tuning",
    project_name="dnn_optimization"
)

# Perform search using all available GPUs
tuner.search(train_X, train_y, validation_data=(test_X, test_y), epochs=3000, batch_size=BATCH)

# Retrieve best models
best_models = tuner.get_best_models(num_models=10)
with open("best_models_summary.txt", "w") as f:
    for i, model in enumerate(best_models):
        model.save(f"best_model_{i}.h5", save_format='h5')
        f.write(f"Model {i} Summary:\n")
        model.summary(print_fn=lambda x: f.write(x + "\n"))
        f.write("\n--------------------------\n")

print("Top 10 models saved based on validation loss. Model architectures saved in best_models_summary.txt.")
