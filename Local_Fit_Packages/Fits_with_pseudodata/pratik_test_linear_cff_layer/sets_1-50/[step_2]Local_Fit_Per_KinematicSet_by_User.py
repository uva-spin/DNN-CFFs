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

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
import joblib

from BHDVCS_tf_modified import *
from user_inputs import *
from DNN_model import *

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")

# Load and prep base data
df = pd.read_csv(initial_data_file)
df['set'] = df['set'].astype(int)
df = df.rename(columns={"sigmaF": "errF"})
df = df[df["F"] != 0]

scratch_path = str(scratch_path) + '/'
create_folders(scratch_path + 'Replica_Cross_Sections')

modify_LR = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=modify_LR_factor, patience=modify_LR_patience, mode='auto'
)
EarlyStop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=EarlyStop_patience, restore_best_weights=True
)

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

def GenerateReplicaData(src_df):
    out = {
        'k': src_df['k'].to_numpy(),
        'QQ': src_df['QQ'].to_numpy(),
        'x_b': src_df['x_b'].to_numpy(),
        't': src_df['t'].to_numpy(),
        'phi_x': src_df['phi_x'].to_numpy(),
        'True_F': src_df['F'].to_numpy(),
        'errF': np.abs(src_df['errF'].to_numpy()),
        'dvcs': src_df['dvcs'].to_numpy(),
    }
    tempF = out['True_F']
    tempFerr = out['errF']
    # Vectorized positivity resampling
    ReplicaF = np.random.normal(loc=tempF, scale=tempFerr)
    neg = ReplicaF < 0
    while np.any(neg):
        ReplicaF[neg] = np.random.normal(loc=tempF[neg], scale=tempFerr[neg])
        neg = ReplicaF < 0
    out['F'] = ReplicaF
    return pd.DataFrame(out)

def gen_F_sanity_check(df_1, kinset, replica_id):
    plt.figure(figsize=(10, 6))
    plt.errorbar(df_1['phi_x'], df_1['True_F'], df_1['errF'], fmt='o', label="True_F", color='red', markersize=5)
    plt.plot(df_1['phi_x'], df_1['F'], marker='o', linestyle='', label="Replica_F", color='blue')
    plt.title(f'F vs Phi for Kinematic Set {kinset}')
    plt.xlabel(r'$\phi_x$')
    plt.ylabel('F')
    plt.legend(loc='best', fontsize='small')
    create_folders(scratch_path + f'Replica_Cross_Sections/Kinematic_Set_{kinset}')
    out_file = scratch_path + f'Replica_Cross_Sections/Kinematic_Set_{kinset}/F_vs_Phi_Kinematic_Set_{kinset}_replica_{replica_id}.pdf'
    plt.savefig(out_file)
    plt.close()

def build_inputs(trainX_df, testX_df):
    # Ensure deterministic phi ordering if needed downstream
    # Here we just build the 8-D inputs: 5 raw + 3 scaled(QQ,x_b,t)
    raw_cols = ['QQ', 'x_b', 't', 'phi_x', 'k']
    scaler = make_pipeline(MinMaxScaler())
    scaled_train = scaler.fit_transform(trainX_df[['QQ', 'x_b', 't']].to_numpy())
    scaled_test  = scaler.transform(testX_df[['QQ', 'x_b', 't']].to_numpy())

    X8_train = np.column_stack([trainX_df[raw_cols].to_numpy(), scaled_train]).astype(np.float32)
    X8_test  = np.column_stack([testX_df[raw_cols].to_numpy(),  scaled_test]).astype(np.float32)
    return X8_train, X8_test, scaler

def run_replica(kinset, replica_number, xdf):
    tempdf = GenerateReplicaData(xdf)

    # Optional sanity plot (keep if you want visuals per replica)
    gen_F_sanity_check(tempdf, kinset, replica_number)

    trainKin, testKin, trainY, testY, trainYerr, testYerr = split_data(
        tempdf[['QQ', 'x_b', 't', 'phi_x', 'k']], tempdf['F'], tempdf['errF'], split=0.1
    )

    # Build 8-D inputs with a scaler fit on train only
    X8_train, X8_test, kins_scaler = build_inputs(trainKin, testKin)

    # Model outputs log-normalized F; targets must be in the same space
    k_val = float(trainKin['k'].iloc[0])

    def y_transform(y_np):
        y_np = np.asarray(y_np, dtype=np.float32)
        if k_val == 5.75:
            return np.log1p(y_np)
        return np.log10(np.maximum(y_np, 1e-12))

    y_train = y_transform(trainY.values)
    y_test  = y_transform(testY.values)

    tfModel = DNNmodel(k_value=k_val)  # uses 8-D inputs and applies the same log in-graph

    history = tfModel.fit(
        X8_train, y_train,
        validation_data=(X8_test, y_test),
        epochs=EPOCHS, callbacks=[modify_LR],
        batch_size=BATCH, verbose=2
    )

    # Save artifacts
    set_dir_models  = str(scratch_path) + f'DNNmodels_Kin_Set_{kinset}/'
    set_dir_losses  = str(scratch_path) + f'Losses_Plots_Kin_Set_{kinset}/'
    create_folders(set_dir_models)
    create_folders(set_dir_losses)

    tfModel.save(set_dir_models + f'model{replica_number}.keras')

    # Persist the scaler so inference uses the exact same transform
    joblib.dump(kins_scaler, set_dir_models + f'scaler_replica_{replica_number}.joblib')

    # Loss plots
    plt.figure(1, figsize=(12, 5))
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history.get('val_loss', []), label='Val. loss')
    plt.title(f'Losses for Kinematic Set {kinset}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(set_dir_losses + f'loss_plots_{replica_number}.pdf')
    plt.close()

# Run across sets
for set_number in kinematic_sets:
    print(f"Running for kinematic set {set_number}")
    kin_df = df[df['set'] == int(set_number)].reset_index(drop=True)

    if len(kin_df) < 10:
        print(f"Skipping set {set_number} (only {len(kin_df)} data points)")
        continue

    create_folders(str(scratch_path) + f'DNNmodels_Kin_Set_{set_number}')
    create_folders(str(scratch_path) + f'Losses_Plots_Kin_Set_{set_number}')

    start_time = datetime.now()
    replica_id = sys.argv[1]  # for SLURM array
    run_replica(set_number, replica_id, kin_df)
    end_time = datetime.now()
    elapsed = end_time - start_time

    time_file = 'time_taken.txt'
    if not os.path.exists(time_file):
        with open(time_file, 'w') as file:
            file.write("Time taken for set number and replica\n")
    with open(time_file, "a") as f:
        f.write(f"Kinematic set {set_number}, replica {replica_id} | Start: {start_time} | End: {end_time} | Elapsed: {elapsed}\n")

    print(f"Completed running for kinematic set {set_number}")
