import tensorflow as tf
import numpy as np
from definitions import TotalFLayer, ModKM15_CFFs, compute_quantities
from user_inputs import *


class FixedCFFsLayer(tf.keras.layers.Layer):
    """
    Custom layer that calculates fixed CFFs (ReE, ReHt, dvcs) from KM15.
    These CFFs are non-trainable and calculated based on kinematics.
    """
    def __init__(self, **kwargs):
        super(FixedCFFsLayer, self).__init__(**kwargs)
        self.trainable = False  # This layer is not trainable

    def call(self, inputs):
        """
        inputs: [QQ, x_b, t, phi, k] - shape (batch_size, 5)
        Returns: [ReE, ReHt, dvcs] - shape (batch_size, 3)
        """
        QQ, x_b, t, phi, k = tf.split(inputs, num_or_size_splits=5, axis=1)
        
        # Convert to numpy for KM15 calculation (since it uses scipy.integrate.quad)
        def compute_fixed_cffs(QQ_np, x_b_np, t_np, phi_np, k_np):
            batch_size = QQ_np.shape[0]
            ReE_list = []
            ReHt_list = []
            dvcs_list = []
            
            for i in range(batch_size):
                # Calculate KM15 CFFs
                ReH_KM15, ImH_KM15, ReE_KM15, ReHt_KM15, ImHt_KM15, ReEt_KM15 = ModKM15_CFFs(
                    float(QQ_np[i, 0]), 
                    float(x_b_np[i, 0]), 
                    float(t_np[i, 0])
                )
                
                # Calculate DVCS term using compute_quantities
                phi_rad = np.deg2rad(float(phi_np[i, 0]))
                _, _, DVCS_term, _ = compute_quantities(
                    float(k_np[i, 0]), 
                    float(QQ_np[i, 0]), 
                    float(x_b_np[i, 0]), 
                    float(t_np[i, 0]), 
                    phi_rad,
                    ReH_KM15, ReHt_KM15, ReE_KM15, ReEt_KM15, 
                    ImH_KM15, ImHt_KM15, 0.0, 0.0
                )
                dvcs_val = float(np.asarray(DVCS_term).squeeze())
                
                ReE_list.append(float(ReE_KM15))
                ReHt_list.append(float(ReHt_KM15))
                dvcs_list.append(dvcs_val)
            
            # Stack into arrays with shape (batch_size,)
            ReE_arr = np.array(ReE_list, dtype=np.float32)
            ReHt_arr = np.array(ReHt_list, dtype=np.float32)
            dvcs_arr = np.array(dvcs_list, dtype=np.float32)
            
            # Stack into shape (batch_size, 3)
            result = np.stack([ReE_arr, ReHt_arr, dvcs_arr], axis=1)
            return result
        
        # Use tf.py_function to call Python function
        fixed_cffs_np = tf.py_function(
            func=compute_fixed_cffs,
            inp=[QQ, x_b, t, phi, k],
            Tout=tf.float32
        )
        
        # Ensure proper shape
        fixed_cffs_np.set_shape([None, 3])
        
        # Stop gradients to ensure these values don't receive gradients during backpropagation
        return tf.stop_gradient(fixed_cffs_np)


def DNNmodel(train_data=None, learning_rate=1e-2):
    """
    Modified DNN model where:
    - Only ReH is predicted by the DNN (output size = 1)
    - ReE, ReHt, and dvcs are calculated from KM15 (fixed, non-trainable)
    """
    # Better weight initialization
    initializer = tf.keras.initializers.HeNormal(seed=42)

    inputs = tf.keras.Input(shape=(5,), name='input_layer')

    norm_layer = tf.keras.layers.Normalization(axis=-1, name="input_normalization")
    x = norm_layer(inputs)

    QQ, x_b, t, phi, k = tf.split(x, num_or_size_splits=5, axis=1)

    # Enhanced kinematics with derived features
    kinematics = tf.keras.layers.concatenate([QQ, x_b, t])
    
    # Enhanced architecture with increased capacity
    x1 = tf.keras.layers.Dense(60, activation="tanh", kernel_initializer=initializer)(kinematics)
    
    x2 = tf.keras.layers.Dense(50, activation="tanh", kernel_initializer=initializer)(x1)
    
    x3 = tf.keras.layers.Dense(40, activation="tanh", kernel_initializer=initializer)(x2)
    
    x4 = tf.keras.layers.Dense(30, activation="tanh", kernel_initializer=initializer)(x3)
    
    # Only ReH is predicted by the DNN (output size changed from 4 to 1)
    ReH_output = tf.keras.layers.Dense(1, activation="linear", kernel_initializer=initializer,
                                       name='ReH_output')(x4)
    
    # Calculate fixed CFFs (ReE, ReHt, dvcs) from KM15
    fixed_cffs = FixedCFFsLayer(name='FixedCFFsLayer')(inputs)  # Use original inputs (not normalized)
    
    # Concatenate ReH (from DNN) with fixed CFFs: [ReH, ReE, ReHt, dvcs]
    all_cffs = tf.keras.layers.concatenate([ReH_output, fixed_cffs], axis=1, name='cff_output_layer')

    # Concatenate original inputs (not normalized) with all CFFs for TotalFLayer
    # TotalFLayer expects: [QQ, x_b, t, phi, k, ReH, ReE, ReHt, dvcs]
    # Note: Use original inputs (not normalized x) because physics calculations need actual values
    total_FInputs = tf.keras.layers.concatenate([inputs, all_cffs], axis=1)
    TotalF = TotalFLayer(name='TotalFLayer')(total_FInputs)

    tfModel = tf.keras.Model(inputs=inputs, outputs=TotalF, name="enhanced_tfmodel_single_cff")
    
    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        alpha=1e-4
    )
    
    tfModel.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['mae']
    )

    if train_data is not None:
        norm_layer.adapt(train_data)

    return tfModel
