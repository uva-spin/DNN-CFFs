import numpy as np
import pandas as pd
import tensorflow as tf
from BHDVCS_tf_modified import *
import matplotlib.pyplot as plt
import argparse
import os
import sys


def create_folders(folder_name):
    try:
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    except FileExistsError:
        print(f"Folder '{folder_name}' already exists!")

        

create_folders('DNNmodels')
create_folders('Losses_CSVs')
create_folders('Losses_Plots')


data_file = 'PseudoData_from_the_Basic_Model.csv'
df = pd.read_csv(data_file, dtype=np.float64)
df = df.rename(columns={"sigmaF": "errF"})

#### User's inputs ####
Learning_Rate = 0.0001
EPOCHS = 100
EarlyStop_patience = 1000
modify_LR_patience = 400
modify_LR_factor = 0.9
NUM_REPLICAS =100
num_hidden_layers = 2
num_nodes = 300
activation_function = "relu"

print(sys.argv)
# Argument Parsing
parser = argparse.ArgumentParser(description='LMIFIT model configuration')
parser.add_argument('replica_number', type=int, help='Replica number')
parser.add_argument('num_nodes', type=int, help='Number of nodes')
parser.add_argument('learning_rate', type=float, help='Learning rate')
parser.add_argument('activation_function', type=str, help='Activation function')
parser.add_argument('EarlyStop_patience', type=int, help='Early stopping patience')
parser.add_argument('modify_lr_patience', type=float, help='Modify learning rate patience')
parser.add_argument('modify_lr_factor', type=float, help='Modify learning rate factor')
parser.add_argument('num_hidden_layers', type=int, help='Number of hidden layers')
args = parser.parse_args()
print(args)
# Using the parsed arguments
num_nodes = args.num_nodes
learning_rate = args.learning_rate
activation_function = args.activation_function
EarlyStop_patience = args.EarlyStop_patience
modify_LR_patience = args.modify_lr_patience
modify_LR_factor = args.modify_lr_factor
num_hidden_layers = args.num_hidden_layers
replica_number = args.replica_number
modify_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=modify_LR_factor,patience=modify_LR_patience,mode='auto')
EarlyStop = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=EarlyStop_patience)

def chisquare(y, yhat, err):
    return np.sum(((y - yhat)/err)**2)

def split_data(X,y,yerr,split=0.1):
  temp =np.random.choice(list(range(len(y))), size=int(len(y)*split), replace = False)

  test_X = pd.DataFrame.from_dict({k: v[temp] for k,v in X.items()})
  train_X = pd.DataFrame.from_dict({k: v.drop(temp) for k,v in X.items()})

  test_y = y[temp]
  train_y = y.drop(temp)

  test_yerr = yerr[temp]
  train_yerr = yerr.drop(temp)

  return train_X, test_X, train_y, test_y, train_yerr, test_yerr


####### Here we define a function that can sample F within sigmaF ###
def GenerateReplicaData(df):
    pseudodata_df = {'k': [],
                     'QQ': [],
                     'x_b': [],
                     't': [],
                     'phi_x': [],
                     'F':[],
                     'errF':[]}
    #pseudodata_df = pd.DataFrame(pseudodata_df)
    pseudodata_df['k'] = df['k']
    pseudodata_df['QQ'] = df['QQ']
    pseudodata_df['x_b'] = df['x_b']
    pseudodata_df['t'] = df['t']
    pseudodata_df['phi_x']= df['phi_x']
    pseudodata_df['errF']= df['errF']
    pseudodata_df['dvcs']= df['dvcs']
    tempF = np.array(df['F'])
    tempFerr = np.abs(np.array(df['errF'])) ## Had to do abs due to a run-time error
    ReplicaF = np.random.normal(loc=tempF, scale=tempFerr)
    pseudodata_df['F']=ReplicaF
    return pd.DataFrame(pseudodata_df)



def DNNmodel():
    initializer_seeds = [np.random.randint(0, 10000) for _ in range(num_hidden_layers + 1)]
    inputs = tf.keras.Input(shape=(5), name='input_layer')
    QQ, x_b, t, phi, k = tf.split(inputs, num_or_size_splits=5, axis=1)
    kinematics = tf.keras.layers.concatenate([QQ, x_b, t], axis=1, name='kinematics_concat')
    x = kinematics
    for i in range(num_hidden_layers):
        initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=initializer_seeds[i])
        x = tf.keras.layers.Dense(num_nodes, activation=activation_function, kernel_initializer=initializer, name=f'hidden_layer_{i+1}')(x)
    final_initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=initializer_seeds[-1])
    cff_output = tf.keras.layers.Dense(4, activation="linear", kernel_initializer=final_initializer, name='cff_output_layer')(x)
    total_FInputs = tf.keras.layers.concatenate([inputs, cff_output], axis=1, name='combined_input_cff')
    TotalF = TotalFLayer(name='total_F_layer')(total_FInputs)
    tfModel = tf.keras.Model(inputs=inputs, outputs=TotalF, name="tfmodel")
    tfModel.compile(optimizer=tf.keras.optimizers.Adam(Learning_Rate), loss=tf.keras.losses.MeanSquaredError())
    return tfModel

    

def run_replica(replica_number):
    tempdf=GenerateReplicaData(df)
    ### If you want to save the replica uncoment the following line with the proper folder name ####
    #tempdf.to_csv(str(Repl_Folder)+'/rep'+str(replica_number)+'.csv')
    trainKin, testKin, trainY, testY, trainYerr, testYerr = split_data(tempdf[['QQ', 'x_b', 't','phi_x', 'k']],tempdf['F'],tempdf['errF'],split =0.1)
    #print(trainKin) , 'dvcs' , 'phi_x', 'k', 'QQ', 'x_b', 't'
    #models = Models()
    tfModel = DNNmodel()
#     tfModel.compile(optimizer = tf.keras.optimizers.Adam(Learning_Rate),
#         loss = tf.keras.losses.MeanSquaredError())
    #, callbacks=[modify_LR,EarlyStop]
    history = tfModel.fit(trainKin, trainY, validation_data=(testKin, testY), epochs=EPOCHS, callbacks=[modify_LR], batch_size=300, verbose=2)
    tfModel.save('DNNmodels/'+'model' + str(replica_number) + '.h5', save_format='h5')
    tempdf = pd.DataFrame()
    tempdf["Train_Loss"] = history.history['loss'][-100:]
    tempdf["Val_Loss"] = history.history['val_loss'][-100:]
    tempdf.to_csv('Losses_CSVs/'+'reploss_'+str(replica_number)+'.csv')
    plt.figure(1)
    plt.plot(history.history['loss'])
    #plt.ylim([0,0.01])
    plt.savefig('Losses_Plots/'+'train_loss'+str(replica_number)+'.pdf')
    plt.figure(2)
    plt.plot(history.history['val_loss'])
    plt.savefig('Losses_Plots/'+'val_loss'+str(replica_number)+'.pdf')

    
###### Running Jobs on Rivanna: Comment the following lines and uncomment the run_replica(), uncomment replica_number = sys.argv[1] and comment replica_number = i in the 'def run_replica()'  
#replica_number = int(sys.argv[1])
run_replica(replica_number)
