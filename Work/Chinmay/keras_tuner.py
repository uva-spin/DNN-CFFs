import tensorflow as tf
import keras_tuner as kt
import numpy as np
import pandas as pd
from BHDVCS_tf import TotalFLayer
data_file = 'pseudoKM15.csv'

df = pd.read_csv(data_file, dtype=np.float64)
df = df.rename(columns={"sigmaF": "errF"})
df = df.reset_index()

  # Changed splitting
def split_data(Kinematics,output,split=0.1):
  temp =np.random.choice(list(range(len(output))), size=int(len(output)*split), replace = False)

  test_X = pd.DataFrame.from_dict({k: v[temp] for k,v in Kinematics.items()})
  train_X = pd.DataFrame.from_dict({k: v.drop(temp) for k,v in Kinematics.items()})

  test_y = output[temp]
  train_y = output.drop(temp)

  return train_X, test_X, train_y, test_y
  
trainKin, testKin, trainOut, testOut = split_data(df[['phi_x', 'k', 'QQ', 'x_b', 't', 'dvcs']],df['F'],split =0.1)

def tf_model1(hp):
      layer_1 = hp.Int("layer_1", min_value = 50, max_value = 1000, step = 50)
      layer_2 = hp.Int("layer_2", min_value = 50, max_value = 1000, step = 50)
      InitialLearningRate = hp.Float("InitialLearningRate", min_value = .000001, max_value = 1, step = .001)
      DecayRate = hp.Float("DecayRate", min_value = .01, max_value = 1, step = .1)
      
      
      initializer = tf.keras.initializers.HeNormal()
      #### QQ, x_b, t, phi, k ####
      inputs = tf.keras.Input(shape=(5))

      QQ, x_b, t, phi, k = tf.split(inputs, num_or_size_splits=5, axis=1)
      kinematics = tf.keras.layers.concatenate([QQ, x_b, t], axis=1)
      x1 = tf.keras.layers.Dense(layer_1, activation="tanh", kernel_initializer=initializer)(kinematics)#tune the "tanh" as well add options 
      x2 = tf.keras.layers.Dense(layer_2, activation="tanh", kernel_initializer=initializer)(x1)
      outputs = tf.keras.layers.Dense(4, activation="linear", kernel_initializer=initializer)(x2)
      #### QQ, x_b, t, phi, k, cffs ####
      total_FInputs = tf.keras.layers.concatenate([inputs, outputs], axis=1)

      TotalF = TotalFLayer()(total_FInputs) # get rid of f1 and f2

      tfModel = tf.keras.Model(inputs=inputs, outputs = TotalF, name="tfmodel")

      lr = tf.keras.optimizers.schedules.ExponentialDecay(
          InitialLearningRate, 1000, DecayRate, staircase=False, name=None
      )

      tfModel.compile(
          optimizer = tf.keras.optimizers.Adam(lr),
          loss = tf.keras.losses.MeanSquaredError()
      )

      return tfModel
  
tuner = kt.Hyperband(tf_model1, objective = "val_accuracy", max_epochs = 100, factor =3)

stop_early = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', patience=5)

tuner.search(trainKin,trainOut,validation_split = .2, callbacks = [stop_early])

best = tuner.get_best_hyperparameters(num_trials = 1)[0]
print(best)