import numpy as np
import pandas as pd
import tensorflow as tf
from BHDVCS_tf_modified import *
import matplotlib.pyplot as plt
import os
import sys


def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")
        

create_folders('DNNmodels')
create_folders('Losses_CSVs')
create_folders('Losses_Plots')
create_folders('Replica_Results')
create_folders('Comparison_Plots')

data_file = 'PseudoData_from_the_Basic_Model.csv'
df = pd.read_csv(data_file, dtype=np.float64)
df = df.rename(columns={"sigmaF": "errF"})



#### User's inputs ####
Learning_Rate = 0.0001
EPOCHS = 5
modify_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.9,patience=400,mode='auto')
EarlyStop = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=1000)
NUM_REPLICAS = 2

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

#GenerateReplicaData(df)



Learning_Rate = 0.0001
def DNNmodel():
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=None)
    #### QQ, x_b, t, phi, k ####
    inputs = tf.keras.Input(shape=(5))
    QQ, x_b, t, phi, k = tf.split(inputs, num_or_size_splits=5, axis=1)
    #kinematics = tf.keras.layers.concatenate([QQ, x_b, t], axis=1)
    #QQ, x_b, t, phi, k = tf.split(inputs, num_or_size_splits=5)
    kinematics = tf.keras.layers.concatenate([QQ, x_b, t])
    x1 = tf.keras.layers.Dense(100, activation="tanh", kernel_initializer=initializer)(kinematics)
    x2 = tf.keras.layers.Dense(100, activation="tanh", kernel_initializer=initializer)(x1)
    outputs = tf.keras.layers.Dense(4, activation="linear", kernel_initializer=initializer)(x2)
    #### QQ, x_b, t, phi, k, cffs ####
    total_FInputs = tf.keras.layers.concatenate([inputs, outputs], axis=1)
    TotalF = TotalFLayer()(total_FInputs) # get rid of f1 and f2
    #tfModel = tf.keras.Model(inputs=inputs, outputs = TotalF, name="tfmodel")
    tfModel = tf.keras.Model(inputs=inputs, outputs = TotalF)
    tfModel.compile(
        optimizer = tf.keras.optimizers.Adam(Learning_Rate),
        loss = tf.keras.losses.MeanSquaredError()
    )
    return tfModel

# testmodel = DNNmodel()
# testmodel.summary()

def calc_yhat(model, X):
    return model.predict(X)


def GenerateReplicaResults(df,model):
    pseudodata_df = {'k': [],
                     'QQ': [],
                     'x_b': [],
                     't': [],
                     'phi_x': [],
                     'F':[],
                     'errF':[],                     
                     'ReH': [],
                     'ReE': [],
                     'ReHt': [],
                     'dvcs': []}
    #pseudodata_df = pd.DataFrame(pseudodata_df)
    tempX = df[['QQ', 'x_b', 't','phi_x', 'k']]
    PredictedCFFs = np.array(tf.keras.backend.function(model.layers[0].input, model.layers[5].output)(tempX))
    PredictedFs = np.array(tf.keras.backend.function(model.layers[0].input, model.layers[7].output)(tempX))
    pseudodata_df['k'] = df['k']
    pseudodata_df['QQ'] = df['QQ']
    pseudodata_df['x_b'] = df['x_b']
    pseudodata_df['t'] = df['t']
    pseudodata_df['phi_x']= df['phi_x']
    pseudodata_df['errF']= df['errF']
    pseudodata_df['dvcs']= df['dvcs']
    pseudodata_df['F']= list(PredictedFs.flatten())
    pseudodata_df['ReH'] = list(PredictedCFFs[:, 0])
    pseudodata_df['ReE'] = list(PredictedCFFs[:, 1])
    pseudodata_df['ReHt'] = list(PredictedCFFs[:, 2])
    pseudodata_df['dvcs'] = list(PredictedCFFs[:, 3])
    return pd.DataFrame(pseudodata_df)

def run_replica(i):
    #replica_number = sys.argv[1]
    replica_number = i
    tempdf=GenerateReplicaData(df)
    ### If you want to save the replica uncoment the following line with the proper folder name ####
    #tempdf.to_csv(str(Repl_Folder)+'/rep'+str(replica_number)+'.csv')
    trainKin, testKin, trainY, testY, trainYerr, testYerr = split_data(tempdf[['QQ', 'x_b', 't','phi_x', 'k']],tempdf['F'],tempdf['errF'],split =0.1)
    tfModel = DNNmodel()
#         loss = tf.keras.losses.MeanSquaredError())
    #, callbacks=[modify_LR,EarlyStop]
    history = tfModel.fit(trainKin, trainY, validation_data=(testKin, testY), epochs=EPOCHS, callbacks=[modify_LR], batch_size=300, verbose=2)
    tfModel.save('DNNmodels/'+'model' + str(replica_number) + '.h5', save_format='h5')
    tempX = df[['QQ', 'x_b', 't','phi_x', 'k']]
    ########################################################################################################
    ######## Here I calculate the predictions for the entire original df file's kinematics ##################
    PredictedCFFs = np.array(tf.keras.backend.function(tfModel.layers[0].input, tfModel.layers[5].output)(tempX))
    PredictedFs = np.array(tf.keras.backend.function(tfModel.layers[0].input, tfModel.layers[7].output)(tempX))
    ########################################################################################################
    ### Preparing the resultant dataframe with the trained model ###
    replicaResultsdf = GenerateReplicaResults(df,tfModel)
    replicaResultsdf.to_csv('Replica_Results/rep_result'+str(replica_number)+'.csv')
    ### Preparing the train/validation loss plots ###
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
    ### Preparing the resultant comparison plots for F,and CFFs ###
    org_file = 'PseudoData_from_the_Basic_Model.csv'
    rep_file = 'Replica_Results/rep_result0.csv'
    org_df=pd.read_csv(org_file, dtype=np.float64)
    rep_df=pd.read_csv(rep_file, dtype=np.float64)
    ##################### Important ! here I used '10' because the phi binning is done with the interval of 10 #####
    org_sliced = org_df.loc[org_df['phi_x'] == 10, :]
    rep_sliced = rep_df.loc[rep_df['phi_x'] == 10, :]
    # Slicing columns from the dataframes
    columns_to_compare = ['F', 'ReH', 'ReE', 'ReHt', 'dvcs']
    original_columns = org_sliced[columns_to_compare]
    replica_columns = rep_sliced[columns_to_compare]
    # Plotting comparisons
    for column in columns_to_compare:
        plt.figure(figsize=(8, 5))
        plt.plot(original_columns[column],'.', label='Original', marker='o')
        plt.plot(replica_columns[column],'.', label='Replica', marker='x')
        plt.title(f'Comparison of {column} between Original and Replica')
        plt.xlabel('Index')
        plt.ylabel(column)
        plt.legend()
        #plt.show()
        plt.savefig(f'Comparison_Plots/{column}_comp_org_with_rep_'+str(replica_number)+'.pdf')
        plt.close()

    
###### Running Jobs on Rivanna: Comment the following lines and uncomment the run_replica(), uncomment replica_number = sys.argv[1] and comment replica_number = i in the 'def run_replica()'  
for i in range(0,NUM_REPLICAS):
    run_replica(i)
#run_reppica()