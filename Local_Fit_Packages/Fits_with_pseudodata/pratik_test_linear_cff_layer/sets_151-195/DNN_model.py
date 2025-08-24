import tensorflow as tf
from BHDVCS_tf_modified import *
from user_inputs import *

# Model architecture matching the "fast" pattern:
# - Input: 8-D (5 raw + 3 scaled[QQ,x_b,t])
# - Dense stack over scaled features: 100 -> 50 -> 25 -> 10 with ReLU
# - Outputs: 3 linear CFFs + 1 DVCS with softplus
# - Physics layer uses raw 5-D + 4 CFFs to produce TotalF
# - Final output is log-normalized F (log1p if k==5.75, else log10)

def DNNmodel(k_value: float):
    init = tf.keras.initializers.GlorotUniform()

    all_inputs = tf.keras.Input(shape=(8,), name='all_kins')
    # Slice: first 5 raw [QQ,x_b,t,phi_x,k], next 3 scaled [QQ_s,x_b_s,t_s]
    raw = tf.keras.layers.Lambda(lambda x: x[:, 0:5], name='raw_split')(all_inputs)
    scaled = tf.keras.layers.Lambda(lambda x: x[:, 5:8], name='scaled_split')(all_inputs)

    x = tf.keras.layers.Dense(10, activation='tanh', kernel_initializer=init)(scaled)
    x = tf.keras.layers.Dense(8,  activation='tanh', kernel_initializer=init)(x)
    x = tf.keras.layers.Dense(6,  activation='tanh', kernel_initializer=init)(x)
    x = tf.keras.layers.Dense(4,  activation='tanh', kernel_initializer=init)(x)

    cff_123 = tf.keras.layers.Dense(3, activation='linear', kernel_initializer=init, name='cff123')(x)
    dvcs    = tf.keras.layers.Dense(1, activation='linear', kernel_initializer=init, name='dvcs')(x)
    cffs4   = tf.keras.layers.Concatenate(name='cff_output_layer')([cff_123, dvcs])

    # Physics layer: [QQ,x_b,t,phi_x,k,cffs...]
    total_inputs = tf.keras.layers.Concatenate(name='phys_concat')([raw, cffs4])
    TotalF = TotalFLayer(name='TotalFLayer')(total_inputs)

    # Output normalization (match training target)
    def _y_norm(y):
        if k_value == 5.75:
            return tf.math.log1p(y)
        # log10(y) with safety clamp
        y = tf.maximum(y, tf.constant(1e-12, dtype=y.dtype))
        return tf.math.log(y) / tf.math.log(tf.constant(10.0, dtype=y.dtype))

    y_log = tf.keras.layers.Lambda(_y_norm, name='y_log_norm')(TotalF)

    model = tf.keras.Model(inputs=all_inputs, outputs=y_log, name="tfmodel")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(Learning_Rate),
        loss='mae'
    )
    return model
