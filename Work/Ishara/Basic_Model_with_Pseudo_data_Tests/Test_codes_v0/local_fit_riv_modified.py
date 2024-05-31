#################################################
#### Modified Liliet's localfit code    #########
#### ~ Ishara Fernando                  #########
#################################################
import sys
#import ROOT
import numpy as np
import tensorflow as tf
from sklearn.model_selection import RepeatedKFold
from tensorflow_addons.activations import tanhshrink
#sys.path.append('../../')
#from Formulation.BHDVCS_tf_modified import TotalFLayer
from BHDVCS_tf_modified import *
import time
tf.keras.utils.get_custom_objects().update({'tanhshrink': tanhshrink})
import pandas as pd

GPD_MODEL = 'basic'
NUM_OF_SETS = 5


#datafile = 'Basic_Model_pseudo_data_for_Jlab_kinematics.csv'
datafile = 'pseudo_basic_BKM10_Jlab_all_t2.csv'
# df = pd.read_csv(datafile, dtype=np.float64)

# def get_np_set(set):
#     set_cut = "set == " + str(set)
#     df = ROOT.RDF.FromCSV('/home/lc2fc/pseudodata/JLab_kin/pseudo_'+GPD_MODEL+'_BKM10_Jlab_all_t2.csv')
#     npset = df.Filter(set_cut).AsNumpy() # F, phi, errF are jagged arrays
#     return npset

def get_data(datafile):
    df = pd.read_csv(datafile)
    return df
#, dtype=np.float64

def gen_replica(pseudo):
    Farray = np.abs(np.array(pseudo['F'])) # need to take the abs values otherwise random.normal would not work
    varFarray = np.array(pseudo['varF']) 
    #varFarrayprocessed = Farray*varFarray
    # F_rep = np.random.normal(Farray, varFarrayprocessed)
    F_rep = np.random.normal(loc=Farray, scale=varFarray*Farray)
    errF_rep = pseudo['varF'] * F_rep
    
    replica_dict = {'k': pseudo['k'],
                    'QQ': pseudo['QQ'],
                    'xB': pseudo['xB'],
                    't': pseudo['t'], 
                    'phi': pseudo['phi'],
                    'F': F_rep,
                    'errF': errF_rep}   
     #'F': F_rep,
     #'errF': errF_rep
    #return replica_dict
    return pd.DataFrame(replica_dict)


#print(gen_replica(df))

def build_model():
    inputs = tf.keras.Input(shape=(5)) # k, QQ, x_b, t, phi
    k, QQ, xB, t, phi = tf.split(inputs, num_or_size_splits=5, axis=1)
    # Normalization
    QQ_norm = tf.keras.layers.Lambda(lambda x: -1 + 2 * (x / 10))(QQ) 
    xB_norm = tf.keras.layers.Lambda(lambda x: -1 + 2 * (x / 0.8))(xB) 
    t_norm = tf.keras.layers.Lambda(lambda x:  -1 + 2 * ((x + 2) / 2 ))(t) 
    # Concatenate
    kinematics = tf.keras.layers.concatenate([QQ_norm, xB_norm, t_norm], axis=1)
    x1 = tf.keras.layers.Dense(100, activation="linear")(kinematics)
    x2 = tf.keras.layers.Dense(100, activation="tanhshrink")(x1)
    x3 = tf.keras.layers.Dense(100, activation="tanhshrink")(x2)
    x4 = tf.keras.layers.Dense(100, activation="tanh")(x3)
    outputs = tf.keras.layers.Dense(4, activation="linear")(x4)
    #### k, QQ, xB, t, phi, ReH, ReE, ReHt, dvcs ####
    total_FInputs = tf.keras.layers.concatenate([inputs, outputs], axis=1)
    #TotalF = TotalFLayer()(total_FInputs) # get rid of f1 and f2
    TotalF = TotalFLayer(name='TotalFLayer')(total_FInputs)
    tfModel = tf.keras.Model(inputs=inputs, outputs = TotalF)
    tfModel.compile(
        optimizer = tf.keras.optimizers.Adam(0.00025),
        loss = tf.keras.losses.MeanSquaredError()
    )
    return tfModel

def fit_replica(i, set, pseudo):
    # ---- generate replica -----
    replica = gen_replica(pseudo)
    kin = np.dstack((replica['k'], replica['QQ'] , replica['xB'], replica['t'], replica['phi']))
    kin = kin.reshape(kin.shape[1:])
    # ---- model fit ---- 
    rkf = RepeatedKFold(n_splits=9, n_repeats=10, random_state=42)
    for train_index, test_index in rkf.split(kin):
        kin_train, kin_test = kin[train_index], kin[test_index]
        F_train, F_test = replica['F'][train_index], replica['F'][test_index]
    models = build_model()
    models.summary()
    history = models.fit(kin_train, F_train, epochs=1000, batch_size=11, validation_data=(kin_test, F_test))
    # tf.keras.models.save_model(models, 'models/'+GPD_MODEL+'/bs_11_epochs_1k_shuffled_default/set_'+str(set)+'/fit_replica_'+str(i)+'.keras') # need "tf.keras.models.save_model" to save custom layer
    # np.save('models/'+GPD_MODEL+'/bs_11_epochs_1k_shuffled_default/set_'+str(set)+'/history_fit_replica_'+str(i)+'.npy',history.history) 
    tf.keras.models.save_model(models, 'models/'+GPD_MODEL+'/bs_11_epochs_1k_shuffled_default/set_'+str(set)+'/fit_replica_'+str(i)+'.keras') # need "tf.keras.models.save_model" to save custom layer
    np.save('models/'+GPD_MODEL+'/bs_11_epochs_1k_shuffled_default/set_'+str(set)+'/history_fit_replica_'+str(i)+'.npy',history.history) 

REPLICA_NUM = 10

for set in range(1, NUM_OF_SETS+1):
    #pseudo = get_np_set(set)
    pseudo = get_data(datafile)
    start = time.time()
    fit_replica(REPLICA_NUM, set, pseudo)
    print("Run Time: ", time.time() - start)
