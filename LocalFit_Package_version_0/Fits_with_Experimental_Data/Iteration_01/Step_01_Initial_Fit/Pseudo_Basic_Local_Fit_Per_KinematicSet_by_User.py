############################################################################
#####  Written by Ishara Fernando, Ani Venkatapuram                  #######
##############  Revised Date: 10/23/2024    ################################
##### Rivanna usage: Run the following commands on your Rivanna terminal####
## source /home/lba9wf/miniconda3/etc/profile.d/conda.sh         ###########
## conda activate env                                            ###########
## pip3 install --user tensorflow-addons==0.21.0                 ###########
############################################################################
############################################################################
### This can be used to run local fits individually,         ###############
### or in parallel on rivanna                                ###############
### Include following lines to submit parallel jobs on rivanna  ############
###  #SBATCH --array=0-300 (or any number of replicas )         ############
### python (file name) $SLURM_ARRAY_TASK_ID                     ############
### User needs to define the Kinematic Set Number               ############
### That means user will need to copy this code with different j ###########
### Also, user can arrange the range of replicas using the slurm file ######
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
        

#data_file = 'Basic_Model_pseudo_data_for_Jlab_kinematics_with_sampling.csv'
data_file = 'AllJlabData_from_Zulkaida_and_Liliet.csv'
df = pd.read_csv(data_file)
df = df.rename(columns={"sigmaF": "errF"})
# Filter out rows where the "F" column contains zero (Note, this is only for the training not for predictions)
df = df[df["F"] != 0]

## Remember to update the following line
scratch_path = '/scratch/<your-computing-id>/DNN_CFFs/LocalFit_BasicModel_Comparison_100_150/Sampled/'
create_folders(scratch_path+'Replica_Cross_Sections')

#### User's inputs ####
Learning_Rate = 0.001
EPOCHS = 2000
BATCH = 20
EarlyStop_patience = 1000
modify_LR_patience = 400
modify_LR_factor = 0.9

## Here define the Kinematic Set with the parameter j ##
# j = 3 # Previously, we had a single set `j`. Now we will use a list of sets.

# You can modify the following list to include the sets you want to run
# This list can be modified dynamically
kinematic_sets = list(range(1, 5))

modify_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=modify_LR_factor, patience=modify_LR_patience, mode='auto')
EarlyStop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=EarlyStop_patience)


def chisquare(y, yhat, err):
    return np.sum(((y - yhat)/err)**2)

def split_data(X, y, yerr, split=0.1):
    temp = np.random.choice(list(range(len(y))), size=int(len(y) * split), replace=False)

    test_X = pd.DataFrame.from_dict({k: v[temp] for k, v in X.items()})
    train_X = pd.DataFrame.from_dict({k: v.drop(temp) for k, v in X.items()})

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
                     'True_F': [],
                     'F': [],
                     'errF': []}
    pseudodata_df['k'] = df['k']
    pseudodata_df['QQ'] = df['QQ']
    pseudodata_df['x_b'] = df['x_b']
    pseudodata_df['t'] = df['t']
    pseudodata_df['phi_x'] = df['phi_x']
    pseudodata_df['errF'] = df['errF']
    #pseudodata_df['dvcs'] = df['dvcs']
    pseudodata_df['True_F'] = df['F']
    tempF = np.array(df['F'])
    tempFerr = np.abs(np.array(df['errF']))  # Had to do abs due to a run-time error
    ReplicaF = np.random.normal(loc=tempF, scale=tempFerr)
    pseudodata_df['F'] = ReplicaF
    return pd.DataFrame(pseudodata_df)

def gen_F_sanity_check(df_1, kinset, replica_id):
    plt.figure(figsize=(10, 6))
    
    # Plot the original data points
    plt.errorbar(df_1['phi_x'], df_1['True_F'], df_1['errF'], fmt='o', label="True_F", color='red', markersize=5)
    plt.plot(df_1['phi_x'], df_1['F'], marker='o', linestyle='', label="Replica_F", color='blue')

    plt.title(f'F vs Phi for Kinematic Set {kinset}')
    plt.xlabel(r'$\phi_x$')
    plt.ylabel('F')
    plt.legend(loc='best', fontsize='small')
    
    create_folders(scratch_path + 'Replica_Cross_Sections/' + f'Kinematic_Set_{kinset}')
    
    output_file = scratch_path + 'Replica_Cross_Sections/' + f'Kinematic_Set_{kinset}/' + f'F_vs_Phi_Kinematic_Set_{kinset}_replica_{replica_id}.pdf'
    plt.savefig(output_file)
    plt.close()
    

def DNNmodel():
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)
    #### QQ, x_b, t, phi, k ####
    inputs = tf.keras.Input(shape=(5), name='input_layer')
    QQ, x_b, t, phi, k = tf.split(inputs, num_or_size_splits=5, axis=1)
    kinematics = tf.keras.layers.concatenate([QQ, x_b, t])
    x1 = tf.keras.layers.Dense(480, activation="relu6", kernel_initializer=initializer)(kinematics)
    x2 = tf.keras.layers.Dense(320, activation="relu6", kernel_initializer=initializer)(x1)
    x3 = tf.keras.layers.Dense(240, activation="relu6", kernel_initializer=initializer)(x2)
    x4 = tf.keras.layers.Dense(120, activation="relu6", kernel_initializer=initializer)(x3)
    x5 = tf.keras.layers.Dense(32, activation="relu6", kernel_initializer=initializer)(x4)
    outputs = tf.keras.layers.Dense(4, activation="linear", kernel_initializer=initializer, name='cff_output_layer')(x5)
    #### QQ, x_b, t, phi, k, cffs ####
    total_FInputs = tf.keras.layers.concatenate([inputs, outputs], axis=1)
    TotalF = TotalFLayer(name='TotalFLayer')(total_FInputs)  # get rid of f1 and f2
    tfModel = tf.keras.Model(inputs=inputs, outputs=TotalF, name="tfmodel")
    tfModel.compile(
        optimizer=tf.keras.optimizers.Adam(Learning_Rate),
        loss=tf.keras.losses.MeanSquaredError()
    )
    return tfModel


def absolute_residual(tr, prd):
    temp_diff = tr - prd
    temp_abs_diff = np.abs(temp_diff)
    return temp_abs_diff


def run_replica(kinset, i, xdf):
    replica_number = i
    tempdf = GenerateReplicaData(xdf)
    gen_F_sanity_check(tempdf, kinset, replica_number)  
    trainKin, testKin, trainY, testY, trainYerr, testYerr = split_data(tempdf[['QQ', 'x_b', 't', 'phi_x', 'k']],
                                                                       tempdf['F'], tempdf['errF'], split=0.1)

    tfModel = DNNmodel()
    history = tfModel.fit(trainKin, trainY, validation_data=(testKin, testY), epochs=EPOCHS, callbacks=[modify_LR],
                          batch_size=BATCH, verbose=2)

    tfModel.save(str(scratch_path) + 'DNNmodels_Kin_Set_' + str(kinset) + '/' + 'model' + str(replica_number) + '.h5', save_format='h5')

    # Create subplots for loss plots
    plt.figure(1, figsize=(12, 5))
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Val. loss')
    plt.title(f'Losses for Kinematic Set {kinset}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(str(scratch_path) + 'Losses_Plots_Kin_Set_' + str(kinset) + '/' + 'loss_plots_' + str(replica_number) + '.pdf')
    plt.close()


# Load prediction inputs from CSV and run for multiple kinematic sets
for set_number in kinematic_sets:  # Loop through the list of kinematic sets
    print(f"Running for kinematic set {set_number}")
    kin_df = df[df['set'] == int(set_number)]
    kin_df = kin_df.reset_index(drop=True)

    # Create necessary folders for each kinematic set
    create_folders(str(scratch_path) + 'DNNmodels_Kin_Set_' + str(set_number))
    create_folders(str(scratch_path) + 'Losses_Plots_Kin_Set_' + str(set_number))

    replica_id = sys.argv[1]
    run_replica(set_number, replica_id, kin_df)
    print(f"Completed running for kinematic set {set_number}")

    
