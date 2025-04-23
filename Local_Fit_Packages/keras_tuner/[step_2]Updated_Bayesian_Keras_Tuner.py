import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
import random
import os
from BHDVCS_tf_modified import *
from user_inputs import *
from tensorflow.keras.callbacks import EarlyStopping

# Seed everything
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# Enable Multi-GPU Training
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Early stopping to avoid wasting time
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

# Timeout callback (30 minutes per trial)
class TimeLimitCallback(tf.keras.callbacks.Callback):
    def __init__(self, max_seconds):
        self.max_seconds = max_seconds
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = tf.timestamp()

    def on_epoch_end(self, epoch, logs=None):
        if tf.timestamp() - self.start_time > self.max_seconds:
            print("⏱️ Trial stopped: time limit reached.")
            self.model.stop_training = True

# Model builder
def build_model(hp):
    with strategy.scope():
        initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
        inputs = tf.keras.Input(shape=(5,), name='input_layer')
        QQ, x_b, t, phi, k = tf.split(inputs, num_or_size_splits=5, axis=1)
        kinematics = tf.keras.layers.concatenate([QQ, x_b, t])

        # Regularizer choice
        regularizer_choice = hp.Choice("regularizer", ["l1", "l2", "none"])
        if regularizer_choice == "l1":
            regularizer = tf.keras.regularizers.l1(1e-5)
        elif regularizer_choice == "l2":
            regularizer = tf.keras.regularizers.l2(1e-5)
        else:
            regularizer = None

        # Build hidden layers
        x = kinematics
        num_layers = hp.Int("num_layers", 2, 6)
        for i in range(num_layers):
            x = tf.keras.layers.Dense(
                units=hp.Int(f"units_{i}", 64, 512, step=64),
                activation=hp.Choice(f"activation_{i}", ["relu", "relu6", "tanh", "selu", "elu"]),
                kernel_initializer=initializer,
                kernel_regularizer=regularizer
            )(x)

        # Output and total F computation
        outputs = tf.keras.layers.Dense(4, activation="linear", kernel_initializer=initializer, name='cff_output_layer')(x)
        total_FInputs = tf.keras.layers.concatenate([inputs, outputs], axis=1)
        TotalF = TotalFLayer(name='TotalFLayer')(total_FInputs)

        model = tf.keras.Model(inputs=inputs, outputs=TotalF, name="tfmodel")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice("learning_rate", [1e-3, 5e-4, 1e-4, 5e-5])
            ),
            loss=tf.keras.losses.MeanSquaredError()
        )
    return model

# Load and preprocess dataset
df = pd.read_csv(initial_data_file)
df = df.rename(columns={"sigmaF": "errF"})
df = df[df["F"] != 0]
X = df[['QQ', 'x_b', 't', 'phi_x', 'k']]
y = df['F']
yerr = df['errF']

def split_data(X, y, yerr, split=0.1, seed=SEED):
    np.random.seed(seed)
    temp = np.random.choice(list(range(len(y))), size=int(len(y) * split), replace=False)
    test_X = X.iloc[temp]
    train_X = X.drop(temp)
    test_y = y.iloc[temp]
    train_y = y.drop(temp)
    test_yerr = yerr.iloc[temp]
    train_yerr = yerr.drop(temp)
    return train_X, test_X, train_y, test_y, train_yerr, test_yerr

train_X, test_X, train_y, test_y, _, _ = split_data(X, y, yerr)

# Safe fallback batch size if BATCH is undefined
batch_size = BATCH if "BATCH" in globals() else 32

# Run the tuner
tuner = kt.BayesianOptimization(
    build_model,
    objective="val_loss",
    max_trials=50,
    executions_per_trial=1,
    directory="bayesian_tuning_updated",
    project_name="dnn_optimization",
    overwrite=False
)

tuner.search(train_X, train_y,
             validation_data=(test_X, test_y),
             epochs=200,
             batch_size=batch_size,
             callbacks=[early_stop, TimeLimitCallback(1800)])  # 30 min per trial

# Save best models
best_models = tuner.get_best_models(num_models=10)
with open("best_models_summary.txt", "w") as f:
    for i, model in enumerate(best_models):
        model.save(f"best_model_{i}.h5", save_format='h5')
        f.write(f"Model {i} Summary:\n")
        model.summary(print_fn=lambda x: f.write(x + "\n"))
        f.write("\n--------------------------\n")

print("✅ Top 10 models saved. See best_models_summary.txt for architectures.")
