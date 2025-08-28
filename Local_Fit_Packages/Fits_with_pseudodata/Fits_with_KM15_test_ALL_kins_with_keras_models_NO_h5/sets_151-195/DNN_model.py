import tensorflow as tf
from BHDVCS_tf_modified import *
from user_inputs import *


#Model architecture

# def DNNmodel():
#     initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)
#     #### QQ, x_b, t, phi, k ####
#     inputs = tf.keras.Input(shape=(5), name='input_layer')
#     QQ, x_b, t, phi, k = tf.split(inputs, num_or_size_splits=5, axis=1)
#     kinematics = tf.keras.layers.concatenate([QQ, x_b, t])
#     x1 = tf.keras.layers.Dense(100, activation="relu6", kernel_initializer=initializer)(kinematics)
#     x2 = tf.keras.layers.Dense(80, activation="relu6", kernel_initializer=initializer)(x1)
#     x3 = tf.keras.layers.Dense(60, activation="relu6", kernel_initializer=initializer)(x2)
#     x4 = tf.keras.layers.Dense(40, activation="relu6", kernel_initializer=initializer)(x3)
#     x5 = tf.keras.layers.Dense(20, activation="relu6", kernel_initializer=initializer)(x4)
#     outputs = tf.keras.layers.Dense(4, activation="linear", kernel_initializer=initializer, name='cff_output_layer')(x5)
#     #### QQ, x_b, t, phi, k, cffs ####
#     total_FInputs = tf.keras.layers.concatenate([inputs, outputs], axis=1)
#     TotalF = TotalFLayer(name='TotalFLayer')(total_FInputs)  # get rid of f1 and f2
#     tfModel = tf.keras.Model(inputs=inputs, outputs=TotalF, name="tfmodel")
#     tfModel.compile(
#         optimizer=tf.keras.optimizers.Adam(Learning_Rate),
#         loss=tf.keras.losses.MeanSquaredError()
#     )
#     return tfModel


def DNNmodel(train_data=None, learning_rate=1e-1):
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)

    inputs = tf.keras.Input(shape=(5,), name='input_layer')

    norm_layer = tf.keras.layers.Normalization(axis=-1, name="input_normalization")
    x = norm_layer(inputs)

    QQ, x_b, t, phi, k = tf.split(x, num_or_size_splits=5, axis=1)

    kinematics = tf.keras.layers.concatenate([QQ, x_b, t])
    x1 = tf.keras.layers.Dense(100, activation="relu", kernel_initializer=initializer)(kinematics)
    x2 = tf.keras.layers.Dense(80, activation="relu", kernel_initializer=initializer)(x1)
    x3 = tf.keras.layers.Dense(60, activation="relu", kernel_initializer=initializer)(x2)
    x4 = tf.keras.layers.Dense(40, activation="relu", kernel_initializer=initializer)(x3)
    x5 = tf.keras.layers.Dense(20, activation="relu", kernel_initializer=initializer)(x4)
    outputs = tf.keras.layers.Dense(4, activation="linear", kernel_initializer=initializer,
                                    name='cff_output_layer')(x5)


    total_FInputs = tf.keras.layers.concatenate([x, outputs], axis=1)
    TotalF = TotalFLayer(name='TotalFLayer')(total_FInputs)

    tfModel = tf.keras.Model(inputs=inputs, outputs=TotalF, name="tfmodel")
    tfModel.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.MeanSquaredError()
    )


    if train_data is not None:
        norm_layer.adapt(train_data)

    return tfModel
