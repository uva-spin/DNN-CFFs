############################################################################
###############  Written by Ishara Fernando                  ###############
##############  Revised Date: 05/20/2024    ################################
##### Rivanna usage: Run the following commands on your Rivanna terminal####
## source /home/lba9wf/miniconda3/etc/profile.d/conda.sh         ###########
## conda activate env                                            ###########
## pip3 install --user tensorflow-addons==0.21.0                 ###########
############################################################################
############################################################################
### This can be used to run local fits individually,         ###############
### or in parallel on rivanna                                ###############
### Include following lines to submit parallel jobs on rivanna  ############
###  #SBATCH --array=0-196 (or the kinematic sets you want)     ############
### python (file name) $SLURM_ARRAY_TASK_ID                     ############
############################################################################

import numpy as np
import pandas as pd
import tensorflow as tf
from BHDVCS_tf_modified import *
import matplotlib.pyplot as plt
from tensorflow_addons.activations import tanhshrink
tf.keras.utils.get_custom_objects().update({'tanhshrink': tanhshrink})
import os
import sys
from scipy.stats import norm


def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")
        


data_file = 'Filtered_Kinematic_Sets_1_to_10.csv'
#df = pd.read_csv(data_file, dtype=np.float64)
df = pd.read_csv(data_file)
df = df.rename(columns={"sigmaF": "errF"})

## Here provide the name of the folder that you want to save your models, loss_plots, and replica_data ##
test_folder_name = 'Test_Arc_00'
scratch_loction = '/scratch/qzf7nj/DNN_CFFs_test_error_band/Tests/'
scratch_path_folder = str(scratch_loction) + str(test_folder_name)
create_folders(str(scratch_path_folder))
scratch_path = scratch_path_folder+'/'
#scratch_path = '/scratch/cee9hc/DNN_CFFs/Tests/Test_Arc_00/'

#### User's inputs ####
#Learning_Rate = 0.0001
Learning_Rate = 0.001
EPOCHS = 2000
BATCH = 20
EarlyStop_patience = 1000
modify_LR_patience = 400
modify_LR_factor = 0.9

NUM_REPLICAS = 300

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
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=None)
    #### QQ, x_b, t, phi, k ####
    inputs = tf.keras.Input(shape=(5), name='input_layer')
    QQ, x_b, t, phi, k = tf.split(inputs, num_or_size_splits=5, axis=1)
    kinematics = tf.keras.layers.concatenate([QQ, x_b, t])
    x1 = tf.keras.layers.Dense(100, activation="linear", kernel_initializer=initializer)(kinematics)
    x2 = tf.keras.layers.Dense(100, activation="tanhshrink", kernel_initializer=initializer)(x1)
    x3 = tf.keras.layers.Dense(100, activation="tanhshrink", kernel_initializer=initializer)(x2)
    x4 = tf.keras.layers.Dense(100, activation="tanh", kernel_initializer=initializer)(x3)
    outputs = tf.keras.layers.Dense(4, activation="linear", kernel_initializer=initializer, name='cff_output_layer')(x4)
    #### QQ, x_b, t, phi, k, cffs ####
    total_FInputs = tf.keras.layers.concatenate([inputs, outputs], axis=1)
    TotalF = TotalFLayer(name='TotalFLayer')(total_FInputs) # get rid of f1 and f2
    tfModel = tf.keras.Model(inputs=inputs, outputs = TotalF, name="tfmodel")
    tfModel.compile(
        optimizer = tf.keras.optimizers.Adam(Learning_Rate),
        loss = tf.keras.losses.MeanSquaredError()
    )
    return tfModel
    
#def DNNmodel():
#    initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=None)
#    #### QQ, x_b, t, phi, k ####
#    inputs = tf.keras.Input(shape=(5), name='input_layer')
#    QQ, x_b, t, phi, k = tf.split(inputs, num_or_size_splits=5, axis=1)
#    kinematics = tf.keras.layers.concatenate([QQ, x_b, t])
#    x1 = tf.keras.layers.Dense(480, activation="relu", kernel_initializer=initializer)(kinematics)
#    x2 = tf.keras.layers.Dense(320, activation="tanhshrink", kernel_initializer=initializer)(x1)
#    x3 = tf.keras.layers.Dense(32, activation="relu", kernel_initializer=initializer)(x2)
#    x4 = tf.keras.layers.Dense(32, activation="relu", kernel_initializer=initializer)(x3)
#    outputs = tf.keras.layers.Dense(4, activation="linear", kernel_initializer=initializer, name='cff_output_layer')(x4)
#    #### QQ, x_b, t, phi, k, cffs ####
#    total_FInputs = tf.keras.layers.concatenate([inputs, outputs], axis=1)
#    TotalF = TotalFLayer(name='TotalFLayer')(total_FInputs) # get rid of f1 and f2
#    tfModel = tf.keras.Model(inputs=inputs, outputs = TotalF, name="tfmodel")
#    tfModel.compile(
#        optimizer = tf.keras.optimizers.Adam(Learning_Rate),
#        loss = tf.keras.losses.MeanSquaredError()
#    )
#    return tfModel



def calc_yhat(model, X):
    return model.predict(X)
    

def absolute_residual(tr, prd):
    temp_diff = tr - prd
    temp_abs_diff = np.abs(temp_diff) 
    return temp_abs_diff


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
                     'dvcs': [],
                     'AbsRes_ReH':[],
                     'AbsRes_ReE':[],
                     'AbsRes_ReHt':[],
                     'AbsRes_dvcs':[]}
    tempX = df[['QQ', 'x_b', 't','phi_x', 'k']]
    PredictedCFFs = np.array(tf.keras.backend.function(model.get_layer(name='input_layer').input, model.get_layer(name='cff_output_layer').output)(tempX))
    PredictedFs = np.array(tf.keras.backend.function(model.get_layer(name='input_layer').input, model.get_layer(name='TotalFLayer').output)(tempX))
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
    pseudodata_df['AbsRes_ReH'] = list(absolute_residual(df['ReH'],pseudodata_df['ReH']))
    pseudodata_df['AbsRes_ReE'] = list(absolute_residual(df['ReE'],pseudodata_df['ReE']))
    pseudodata_df['AbsRes_ReHt'] = list(absolute_residual(df['ReHt'],pseudodata_df['ReHt']))
    pseudodata_df['AbsRes_dvcs'] = list(absolute_residual(df['dvcs'],pseudodata_df['dvcs']))
    return pd.DataFrame(pseudodata_df)

def plot_F_vs_phi_with_error_bands(df, model, kinset, num_replicas=300):
    """
    Generates the F vs phi_x graph for each kinematic set with error bands based on multiple replicas.
    """
    phi_values = []
    f_predictions = []

    # Get the kinematic data for the set
    tempX = df[['QQ', 'x_b', 't', 'phi_x', 'k']]
    
    # Loop through each replica and generate predictions
    for i in range(num_replicas):
        # Predict F values using the trained model
        PredictedFs = np.array(tf.keras.backend.function(model.get_layer(name='input_layer').input, model.get_layer(name='TotalFLayer').output)(tempX))
        
        # Store the predictions for F and corresponding phi values
        f_predictions.append(PredictedFs.flatten())
        if i == 0:
            phi_values = df['phi_x'].values

    # Convert lists to arrays for easier manipulation
    f_predictions = np.array(f_predictions)  # shape: (num_replicas, num_phi_values)

    # Calculate mean and standard deviation of F predictions at each phi value
    f_mean = np.mean(f_predictions, axis=0)
    f_std = np.std(f_predictions, axis=0)

    # Plot the mean F vs phi with error bands
    plt.figure(figsize=(8, 6))
    plt.plot(phi_values, f_mean, label='Mean F Prediction', color='blue')
    plt.fill_between(phi_values, f_mean - f_std, f_mean + f_std, color='lightblue', alpha=0.5, label='±1 Std Dev')
    plt.xlabel('Phi_x')
    plt.ylabel('F')
    plt.title(f'F vs Phi with Error Bands for Kinematic Set {kinset}')
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(f'F_vs_Phi_with_ErrorBands_Set_{kinset}.png')
    plt.show()


def run_replica(kinset,i,xdf):
    #replica_number = sys.argv[1]   # If you want to use this scrip for job submission, then uncomment this line, 
    #  then comment the following line, and then delete the 'i' in the parenthesis of run_replica(i) function's definition
    replica_number = i
    tempdf = GenerateReplicaData(xdf)
    #str(scratch_path)+'ReplicaData_Kin_Set_'+str(j)
    #tempdf.to_csv('Replica_Data/rep_data' + str(replica_number) + '.csv')
    tempdf.to_csv(str(scratch_path) + 'ReplicaData_Kin_Set_'+str(j) + '/rep_data' + str(replica_number) + '.csv')

    trainKin, testKin, trainY, testY, trainYerr, testYerr = split_data(tempdf[['QQ', 'x_b', 't', 'phi_x', 'k']],
                                                                       tempdf['F'], tempdf['errF'], split=0.1)

    tfModel = DNNmodel()
    history = tfModel.fit(trainKin, trainY, validation_data=(testKin, testY), epochs=EPOCHS, callbacks=[modify_LR],
                          batch_size=BATCH, verbose=2)
    
    tfModel.save(str(scratch_path)+'DNNmodels_Kin_Set_'+ str(kinset)+ '/' + 'model' + str(replica_number) + '.h5', save_format='h5')

    tempX = df[['QQ', 'x_b', 't', 'phi_x', 'k']]

    PredictedCFFs = np.array(tf.keras.backend.function(tfModel.get_layer(name='input_layer').input, tfModel.get_layer(name='cff_output_layer').output)(tempX))
    PredictedFs = np.array(tf.keras.backend.function(tfModel.get_layer(name='input_layer').input, tfModel.get_layer(name='TotalFLayer').output)(tempX))


    # Create subplots for loss plots
    plt.figure(1,figsize=(12, 5))
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Val. loss')
    plt.title('Losses')
    #plt.ylim(0,0.01)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(str(scratch_path)+'Losses_Plots_Kin_Set_'+str(kinset)+'/' + 'loss_plots_' + str(replica_number) + '.pdf')
    plt.close()


def load_FLayer_and_cffLayer(model):
    #config = model.get_config()
    LayerF = tf.keras.models.load_model(model, custom_objects={'TotalFLayer': TotalFLayer})
    #keras.utils.custom_object_scope(custom_objects)
    #get_custom_objects().update({'tanhshrink': tanhshrink})
    LayerCFFs = tf.keras.Model(inputs=LayerF.input, outputs=LayerF.get_layer('cff_output_layer').output)
    return LayerF, LayerCFFs

def predict_cffs_and_f(LayerCFFs, LayerF, inputs):
    cffs = LayerCFFs.predict(inputs)
    f_values = LayerF.predict(inputs)
    return cffs, f_values

import datetime
###### Running Jobs on Rivanna: Comment the following lines and uncomment the run_replica(), uncomment replica_number = sys.argv[1] and comment replica_number = i in the 'def run_replica()'  
create_folders('Comparison_Plots')
#for j in range(1,196):
j = sys.argv[1]
# Load prediction inputs from CSV
set_number = int(j)  # You can specify the desired set number
kin_df = df[df['set'] == int(set_number)]
kin_df = kin_df.reset_index(drop=True)
#print(kin_df)
create_folders(str(scratch_path)+'DNNmodels_Kin_Set_'+str(j))
create_folders(str(scratch_path)+'Losses_Plots_Kin_Set_'+str(j))
create_folders(str(scratch_path)+'ReplicaData_Kin_Set_'+str(j))
print('############# Set Number '+str(set_number)+' ####################')
for i in range(0,NUM_REPLICAS):
    starttime = datetime.datetime.now().replace(microsecond=0)
    run_replica(j,i,kin_df)
    finistime = datetime.datetime.now().replace(microsecond=0)
    print('Completed replica '+str(i)+ ' of Set number '+ str(j))
    print('##################')
    print('The duration for this replica')
    print(finistime - starttime)
# ########## Evaluation ######
# Load models
models_folder = str(scratch_path)+'DNNmodels_Kin_Set_'+str(j)
models = [os.path.join(models_folder, f) for f in os.listdir(models_folder) if f.endswith('.h5')]

# Take only one (or the first) line from grid_df
prediction_inputs = kin_df[kin_df['set'] == set_number].head(1)[['QQ', 'x_b', 't', 'phi_x', 'k']].to_numpy()
set_data = kin_df[kin_df['set'] == set_number].head(1)[['ReH', 'ReE', 'ReHt', 'dvcs']]

#print(len(models))

# Get true values
true_values = set_data.iloc[0].values
f_predictions = []
cff_predictions = []

# Predict CFFs and F values for each model
for model_id in models:
    tempFLayer, tempCFFsLayer = load_FLayer_and_cffLayer(model_id)
    cffs, f_values = predict_cffs_and_f(tempCFFsLayer, tempFLayer, prediction_inputs)
    f_predictions.append(f_values)
    cff_predictions.append(cffs)

# Create subplots in a single figure for CFF histograms with vertical lines
plt.figure(figsize=(15, 10))
cff_labels = ['ReH', 'ReE', 'ReHt', 'dvcs']
for i, cff_label in enumerate(cff_labels):
    plt.subplot(2, 2, i+1)
    data = np.array(cff_predictions)[:, :, i].T.flatten()
    plt.hist(data, bins=20, edgecolor='black', alpha=0.7, color='lightblue')

    mean_value = np.mean(data)
    std_deviation = np.std(data)

    # Plot vertical lines for true value, mean, and bounds for 1-sigma
    plt.axvline(x=true_values[i], color='red', linestyle='--', label='True Value')
    plt.axvline(x=mean_value, color='blue', linestyle='--', label='Mean')
    plt.axvline(x=mean_value - std_deviation, color='green', linestyle='--', label='1-sigma')
    plt.axvline(x=mean_value + std_deviation, color='green', linestyle='--')

    # Fit a Gaussian curve with the correct x-axis limits
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean_value, std_deviation)
    plt.plot(x, p * len(data) * (xmax - xmin) / 20, 'k', linewidth=2)

    plt.title(f'Set {set_number}: {cff_label} Histogram (from Local)\n Mean: {mean_value:.4f}, Std Dev: {std_deviation:.4f}')
    plt.xlabel(cff_label)
    plt.ylabel('Frequency')
    plt.legend()

# Save the figure as a PDF file
output_pdf_path_combined = 'Comparison_Plots/'+'CFFs_CombinedPlots_subplot_kinematic_set_'+ str(j) +'.pdf'
plt.tight_layout()
plt.savefig(output_pdf_path_combined)
plt.close()