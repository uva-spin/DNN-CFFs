import os
import numpy as np
import time
from evaluate import run_set

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
    'model_summary' : False,
    'verbose': 0,
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

if __name__ == "__main__":
    
    folder_name = 'run_1'
    sets = np.arange(20)+1
    num_replicas = 300
    threads = 40

    os.makedirs('data', exist_ok=True)
    os.makedirs(f'data/{folder_name}', exist_ok=True)
    os.makedirs(f'data/{folder_name}/sample', exist_ok=True)
    os.makedirs(f'data/{folder_name}/history', exist_ok=True)

    show_devices = False
    if show_devices:
        print("Visible devices:", tf.config.get_visible_devices())
        print("Intra threads:", tf.config.threading.get_intra_op_parallelism_threads())
        print("Inter threads:", tf.config.threading.get_inter_op_parallelism_threads())
    
    start_time = time.time()

    for set_id in sets:
        run_set(CONFIG, folder_name, set_id, num_replicas, threads)

    end_time = time.time()

    elapsed = int(end_time - start_time)
    hours = elapsed // 3600
    minutes = (elapsed % 3600) // 60
    seconds = elapsed % 60
    print(f'\nTotal runtime: {hours}h {minutes}m {seconds}s')