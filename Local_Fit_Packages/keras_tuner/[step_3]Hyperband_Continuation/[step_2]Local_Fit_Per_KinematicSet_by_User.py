import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
from BHDVCS_tf_modified import *
from user_inputs import *
from tensorflow.keras import layers, models, optimizers

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

df = pd.read_csv(initial_data_file)
df = df.rename(columns={"sigmaF": "errF"})
df = df[df["F"] != 0]
scratch_path = str(scratch_path) + '/'
create_folders(scratch_path + 'Replica_Cross_Sections')

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

def GenerateReplicaData(df):
    pseudodata_df = {col: df[col] for col in ['k', 'QQ', 'x_b', 't', 'phi_x', 'errF', 'dvcs']}
    pseudodata_df['True_F'] = df['F']
    tempF = np.array(df['F'])
    tempFerr = np.abs(np.array(df['errF']))
    while True:
        ReplicaF = np.random.normal(loc=tempF, scale=tempFerr)
        if np.all(ReplicaF > 0):
            break
    pseudodata_df['F'] = ReplicaF
    return pd.DataFrame(pseudodata_df)

def gen_F_sanity_check(df_1, kinset, replica_id, model_index):
    plt.figure(figsize=(10, 6))
    plt.errorbar(df_1['phi_x'], df_1['True_F'], df_1['errF'], fmt='o', label="True_F", color='red', markersize=5)
    plt.plot(df_1['phi_x'], df_1['F'], marker='o', linestyle='', label="Replica_F", color='blue')
    plt.title(f'F vs Phi for Kinematic Set {kinset}')
    plt.xlabel(r'$\phi_x$')
    plt.ylabel('F')
    plt.legend(loc='best', fontsize='small')
    out_folder = scratch_path + f'Replica_Cross_Sections/Kinematic_Set_{kinset}'
    create_folders(out_folder)
    plt.savefig(f'{out_folder}/F_vs_Phi_Kinematic_Set_{kinset}_model_{model_index}_replica_{replica_id}.pdf')
    plt.close()

def build_model_from_hparams(hparams):
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
    inputs = tf.keras.Input(shape=(5,), name='input_layer')
    QQ, x_b, t, phi, k = tf.split(inputs, num_or_size_splits=5, axis=1)
    kinematics = tf.keras.layers.concatenate([QQ, x_b, t])

    x = kinematics
    for i in range(hparams['num_layers']):
        units = hparams.get(f'units_{i}', 64)
        x = tf.keras.layers.Dense(
            units,
            activation=hparams['activation'],
            kernel_initializer=initializer
        )(x)

    outputs = tf.keras.layers.Dense(
        4, activation="linear", kernel_initializer=initializer, name='cff_output_layer'
    )(x)
    total_FInputs = tf.keras.layers.concatenate([inputs, outputs], axis=1)
    TotalF = TotalFLayer(name='TotalFLayer')(total_FInputs)

    model = tf.keras.Model(inputs=inputs, outputs=TotalF, name="tfmodel")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hparams['learning_rate']),
        loss=tf.keras.losses.MeanSquaredError()
    )
    return model


def run_replica(kinset, replica_id, xdf, model_index):
    model_hparams = top_model_hparams[model_index]
    model = build_model_from_hparams(model_hparams)

    tempdf = GenerateReplicaData(xdf)
    gen_F_sanity_check(tempdf, kinset, replica_id, model_index)
    trainKin, testKin, trainY, testY, trainYerr, testYerr = split_data(
        tempdf[['QQ', 'x_b', 't', 'phi_x', 'k']], tempdf['F'], tempdf['errF'], split=0.1)
    
    history = model.fit(trainKin, trainY, validation_data=(testKin, testY),
                        epochs=EPOCHS, callbacks=[modify_LR], batch_size=BATCH, verbose=2)

    model_dir = scratch_path + f'DNNmodels_Kin_Set_{kinset}/'
    loss_dir = scratch_path + f'Losses_Plots_Kin_Set_{kinset}/'
    create_folders(model_dir)
    create_folders(loss_dir)
    
    model.save(f'{model_dir}/model{replica_id}_topmodel{model_index}.h5')
    
    plt.figure(1, figsize=(12, 5))
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Val. loss')
    plt.title(f'Losses for KinSet {kinset}, Model {model_index}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{loss_dir}/loss_plots_{replica_id}_topmodel{model_index}.pdf')
    plt.close()

# MAIN EXECUTION
slurm_id = int(sys.argv[1])
model_index = (slurm_id - 1) // 5
replica_index = (slurm_id - 1) % 5

for set_number in kinematic_sets:
    print(f"Running Kinematic Set {set_number} | Model {model_index} | Replica {replica_index}")
    kin_df = df[df['set'] == int(set_number)].reset_index(drop=True)
    run_replica(set_number, replica_index, kin_df, model_index)
    print(f"Completed Kinematic Set {set_number} | Model {model_index} | Replica {replica_index}")
