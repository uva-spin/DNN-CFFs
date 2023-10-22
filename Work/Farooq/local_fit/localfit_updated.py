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

from sklearn.model_selection import train_test_split

from model_utils import Models

data_file = 'BKM_pseudodata.csv'

df = pd.read_csv(data_file, dtype=np.float64)
df = df.rename(columns={"sigmaF": "errF"})

train, test = map(DvcsData, train_test_split(df, test_size=0.5)) # splitting dataset into training and testing datasets

models = Models()

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0000005, patience=100)

tfModel = models.tf_model1(len(train.Kinematics))
Wsave = tfModel.get_weights()
tfModel.set_weights(Wsave)

tfModel.fit([train.Kinematics, train.XnoCFF], train.sampleY(),
            epochs=10, verbose=1, batch_size=16, callbacks=[early_stopping_callback], 
            validation_data=([test.Kinematics, test.XnoCFF], test.sampleY())) # validation loss

cffs = cffs_from_globalModel(tfModel, train.Kinematics, numHL=2)

df = pd.DataFrame(cffs)

if len(sys.argv) > 1:
    df.to_csv('bySetCFFs' + sys.argv[1] + '.csv')
else:
    df.to_csv('bySetCFFs.csv')

#######################-------Farooq's Edition#######################-------
plt.plot(tfModel.history.history['loss'])
plt.plot(tfModel.history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig('sample_BKM.png')
plt.show()  