#################################################################################
###############  Written by Ishara Fernando                         #############
##############  Revised Date: 05/02/2024           ##############################
#####  This code is written for tensorflow version 2.11.00          #############
#####  replace 'kerastuner' with 'keras_tuner' if you use version 2.13.00 #######
#################################################################################

import numpy as np
import pandas as pd
from BHDVCS_tf_modified import *
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras_tuner.tuners import BayesianOptimization
from sklearn.model_selection import train_test_split
from tensorflow_addons.activations import tanhshrink
tf.keras.utils.get_custom_objects().update({'tanhshrink': tanhshrink})


np.random.seed(42)  # Seed for reproducibility


data_file = 'Basic_Model_pseudo_data_for_Jlab_kinematics.csv'
#df = pd.read_csv(data_file, dtype=np.float64)
tempdf = pd.read_csv(data_file)
tempdf = tempdf.rename(columns={"sigmaF": "errF"})


# Load prediction inputs from CSV
set_number = 'All'  # You can specify the desired set number
#tempdf = tempdf[tempdf['set'] == set_number]
#tempdf = tempdf.reset_index(drop=True)

L1_reg = 10**(-12)

def build_model(hp):
    model = keras.Sequential()
    inputs = tf.keras.Input(shape=(5,))
    QQ, x_b, t, phi, k = tf.split(inputs, num_or_size_splits=5, axis=1)
    kinematics = tf.keras.layers.concatenate([QQ, x_b, t], axis=1)
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=42)  
    for i in range(hp.Int('num_layers', 1, 7)):
        kinematics = tf.keras.layers.Dense(
            units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
            activation=hp.Choice('activation_' + str(i), ['relu', 'relu6', 'tanh', 'tanhshrink', 'selu', 'sigmoid', 'softmax'])
        , kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(kinematics)
    outputs = tf.keras.layers.Dense(4, activation="linear", kernel_initializer=initializer)(kinematics)
    #### QQ, x_b, t, phi, k, cffs ####
    total_FInputs = tf.keras.layers.concatenate([inputs, outputs], axis=1)
    TotalF = TotalFLayer()(total_FInputs) # get rid of f1 and f2
    tfModel = tf.keras.Model(inputs=inputs, outputs = TotalF)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    tfModel.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )
    return tfModel


# Here, BayesianOptimization is used to find the best hyperparameter values by building and evaluating different models
tuner = BayesianOptimization(
    build_model,
    objective='mean_squared_error',
    # objective='val_mean_squared_error',
    max_trials=50,  # The total number of trials (model configurations) to test
    executions_per_trial=1,  # The number of models that should be built and fit for each trial
    directory='my_dir'+str(set_number),
    project_name='keras_tuner_cffs_local_'+str(set_number)
)


def split_data(X,y,yerr,split=0.1):
  temp =np.random.choice(list(range(len(y))), size=int(len(y)*split), replace = False)

  test_X = pd.DataFrame.from_dict({k: v[temp] for k,v in X.items()})
  train_X = pd.DataFrame.from_dict({k: v.drop(temp) for k,v in X.items()})

  test_y = y[temp]
  train_y = y.drop(temp)

  test_yerr = yerr[temp]
  train_yerr = yerr.drop(temp)

  return train_X, test_X, train_y, test_y, train_yerr, test_yerr

   
trainKin, testKin, trainY, testY, trainYerr, testYerr = split_data(tempdf[['QQ', 'x_b', 't', 'phi_x', 'k']], tempdf['F'], tempdf['errF'], split=0.1)


tuner.search(trainKin, trainY, epochs=300, validation_data=(testKin, testY))

# After the search, retrieve the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=10)[0]

hyp_par_dict = best_hps.values
hyp_df = pd.DataFrame(hyp_par_dict.items(), columns=['Hyperparameter','Value'])
hyp_df.to_csv('best_hyperparameters_'+str(set_number)+'.csv', index=False)


