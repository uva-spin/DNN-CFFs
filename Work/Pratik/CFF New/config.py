import os
import numpy as np

# ---- GPU control ----
# Hide all GPUs (CPU-only mode)
os.environ["CUDA_VISIBLE_DEVICES"] = ""  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

# # ---- Threading control ----
# # Limit intra/inter op threads
tf.get_logger().setLevel('ERROR')
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import keras

# ---- Global config variables ----
CONFIG = {
    'show_devices': False,
    'model_summary' : False,
    'verbose': 0,
    'sets': np.arange(20)+1,
    'replicas': 300,
    'learning_rate': 1e-1,
    'modify_LR_factor': 0.8,
    'modify_LR_patience': 30,
    'minimum_LR': 1e-5,
    'early_stop_patience': 30,
    'layers': [10, 50, 100, 50, 10],
    'activation': 'relu', 
    'initializer': keras.initializers.glorot_uniform(),
    'epochs' : 500,
    'batch_size' : 2
}