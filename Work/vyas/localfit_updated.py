import numpy as np
import pandas as pd

from BHDVCS_tf import DvcsData
from BHDVCS_tf import cffs_from_globalModel
from BHDVCS_tf import F2VsPhi as F2VsPhitf
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt

import sys
from scipy.stats import chisquare

from utils import Models

data_file = 'BKM_pseudodata.csv'

df = pd.read_csv(data_file, dtype=np.float64)
df = df.rename(columns={"sigmaF": "errF"})

data = DvcsData(df)

models = Models()

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0000005, patience=100)

tfModel = models.tf_model1() 
Wsave = tfModel.get_weights()
tfModel.set_weights(Wsave)

tfModel.fit([data.Kinematics[['QQ', 'x_b', 't']], data.XnoCFF], data.sampleY(), # one replica of samples from F vals
            epochs=100, verbose=1, batch_size=16, callbacks=[early_stopping_callback])

cffs = cffs_from_globalModel(tfModel, data.Kinematics[['QQ', 'x_b', 't']], numHL=2)

df = pd.DataFrame(cffs)

if len(sys.argv) > 1:
    df.to_csv('bySetCFFs' + sys.argv[1] + '.csv')
else:
    df.to_csv('bySetCFFs.csv')