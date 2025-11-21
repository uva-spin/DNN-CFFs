import tensorflow as tf
from BHDVCS_tf_modified import *
from user_inputs import *
from tensorflow.keras import regularizers


def create_standard_cff_dnn(kinematics, initializer, name_prefix):
    """
    Creates the baseline DNN architecture for predicting a single CFF.

    Args:
        kinematics: Input tensor containing kinematic variables (QQ, x_b, t, phi, k)
        initializer: Weight initializer for the layers
        name_prefix: Prefix for layer names (e.g., 'ReH', 'ReE', etc.)

    Returns:
        Output tensor representing the predicted CFF value.
    """
    x = tf.keras.layers.Dense(
        100, activation="relu6", kernel_initializer=initializer,
        name=f'{name_prefix}_dense_100'
    )(kinematics)

    x = tf.keras.layers.Dense(
        80, activation="relu6", kernel_initializer=initializer,
        name=f'{name_prefix}_dense_80'
    )(x)

    x = tf.keras.layers.Dense(
        60, activation="relu6", kernel_initializer=initializer,
        name=f'{name_prefix}_dense_60'
    )(x)

    x = tf.keras.layers.Dense(
        40, activation="relu6", kernel_initializer=initializer,
        name=f'{name_prefix}_dense_40'
    )(x)

    x = tf.keras.layers.Dense(
        20, activation="relu6", kernel_initializer=initializer,
        name=f'{name_prefix}_dense_20'
    )(x)

    output = tf.keras.layers.Dense(
        1, activation="relu6", kernel_initializer=initializer,
        name=name_prefix
    )(x)
    return output


def create_ReE_dnn(kinematics, initializer):
    return create_standard_cff_dnn(kinematics, initializer, 'ReE')


def create_ReHt_dnn(kinematics, initializer):
    return create_standard_cff_dnn(kinematics, initializer, 'ReHtilde')


def create_dvcs_dnn(kinematics, initializer):
    return create_standard_cff_dnn(kinematics, initializer, 'dvcs')


def create_ReH_dnn(kinematics, initializer):
    """
    ReH network for the 'thinner_less_layer' trial:
    reduce widths and drop one hidden layer.
    """
    layer_units = [60, 40, 20, 10]

    x = kinematics
    for idx, units in enumerate(layer_units):
        x = tf.keras.layers.Dense(
            units,
            activation="relu6",
            kernel_initializer=initializer,
            name=f'ReH_dense_{units}_layer{idx + 1}'
        )(x)

    output = tf.keras.layers.Dense(
        1, activation="relu6", kernel_initializer=initializer, name='ReH'
    )(x)
    return output


def CrossSection():
    """
    Main model function that creates the cross-section prediction model.
    Uses 4 separate DNNs, one for each CFF (ReH, ReE, ReHtilde, dvcs).
    """
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)

    inputs = tf.keras.Input(shape=(5,), name='input_layer')
    QQ, x_b, t, phi, k = tf.split(inputs, num_or_size_splits=5, axis=1)

    kinematics = tf.keras.layers.concatenate([QQ, x_b, t, phi, k])

    ReH = create_ReH_dnn(kinematics, initializer)
    ReE = create_ReE_dnn(kinematics, initializer)
    ReHtilde = create_ReHt_dnn(kinematics, initializer)
    dvcs = create_dvcs_dnn(kinematics, initializer)

    cff_output_layer = tf.keras.layers.concatenate([ReH, ReE, ReHtilde, dvcs], name='cff_output_layer')

    total_FInputs = tf.keras.layers.concatenate([inputs, cff_output_layer], axis=1)

    TotalF = TotalFLayer(name='TotalFLayer')(total_FInputs)

    tfModel = tf.keras.Model(inputs=inputs, outputs=TotalF, name="tfmodel")
    tfModel.compile(
        optimizer=tf.keras.optimizers.Adam(Learning_Rate),
        loss=tf.keras.losses.MeanSquaredError()
    )
    return tfModel


def DNNmodel():
    return CrossSection()


# def DNNmodel(train_data=None, learning_rate=1e-2):
#     # Better weight initialization
#     initializer = tf.keras.initializers.HeNormal(seed=42)

#     inputs = tf.keras.Input(shape=(5,), name='input_layer')

#     norm_layer = tf.keras.layers.Normalization(axis=-1, name="input_normalization")
#     x = norm_layer(inputs)

#     QQ, x_b, t, phi, k = tf.split(x, num_or_size_splits=5, axis=1)

#     # Enhanced kinematics with derived features
#     kinematics = tf.keras.layers.concatenate([QQ, x_b, t])
    
#     # Enhanced architecture with increased capacity
#     x1 = tf.keras.layers.Dense(
#         60, activation="relu6", kernel_initializer=initializer,
#         kernel_regularizer=regularizers.l2(1e-4), name="dense_60_relu6"
#     )(kinematics)
    
#     x2 = tf.keras.layers.Dense(
#         50, activation="tanh", kernel_initializer=initializer,
#         kernel_regularizer=regularizers.l2(5e-5), name="dense_50_tanh"
#     )(x1)
    
#     x3 = tf.keras.layers.Dense(
#         40, activation="relu6", kernel_initializer=initializer,
#         kernel_regularizer=regularizers.l2(1e-4), name="dense_40_relu6"
#     )(x2)
    
#     x4 = tf.keras.layers.Dense(
#         30, activation="tanh", kernel_initializer=initializer,
#         kernel_regularizer=regularizers.l2(5e-5), name="dense_30_tanh"
#     )(x3)
    
#     x5 = tf.keras.layers.Dense(
#         20, activation="relu6", kernel_initializer=initializer,
#         kernel_regularizer=regularizers.l2(1e-4), name="dense_20_relu6"
#     )(x4)
    
#     outputs = tf.keras.layers.Dense(
#         4, activation="linear", kernel_initializer=initializer,
#         activity_regularizer=regularizers.l2(1e-4), name='cff_output_layer'
#     )(x5)

#     total_FInputs = tf.keras.layers.concatenate([x, outputs], axis=1)
#     TotalF = TotalFLayer(name='TotalFLayer')(total_FInputs)

#     tfModel = tf.keras.Model(inputs=inputs, outputs=TotalF, name="enhanced_tfmodel")
    
#     # Learning rate schedule
#     lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
#         initial_learning_rate=learning_rate,
#         decay_steps=1000,
#         alpha=1e-4
#     )
    
#     tfModel.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
#         loss=tf.keras.losses.MeanSquaredError(),
#         metrics=['mae']
#     )

#     if train_data is not None:
#         norm_layer.adapt(train_data)

#     return tfModel
