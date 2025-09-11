from bkm10 import BKM10
from cff_fit_model import CFF_Fit_Model
import os
import gc
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

def run_replica(CONFIG, set_id, replica_id, kinematics, df_inputs, df_outputs):
    
    model_class = CFF_Fit_Model(
        verbose =         CONFIG['verbose'],
        LR =              CONFIG['learning_rate'],
        mod_LR_factor =   CONFIG['modify_LR_factor'],
        mod_LR_patience = CONFIG['modify_LR_patience'],
        min_LR =          CONFIG['minimum_LR'],
        ES_patience =     CONFIG['early_stop_patience'])
    
    model_class.create_model(
        layers =      CONFIG['layers'],
        activation =  CONFIG['activation'], 
        initializer = CONFIG['initializer'], 
        summary =     CONFIG['model_summary'])

    sample_dsig = np.random.normal(loc=df_outputs['dsig_exp'], scale=df_outputs['dsig_err'])

    while np.any(sample_dsig <= 0):
        mask = sample_dsig <= 0
        sample_dsig[mask] = np.random.normal(loc=df_outputs['dsig_exp'][mask], scale=df_outputs['dsig_err'][mask])

    outs_train = df_outputs.copy()
    outs_train['dsig_exp'] = sample_dsig

    outs_train = shuffle(outs_train)

    kins_train = tf.convert_to_tensor(df_inputs, dtype=tf.float32)
    outs_train = tf.convert_to_tensor(outs_train, dtype=tf.float32)

    history = model_class.fit_model(
        kinematics_class = kinematics, 
        scaled_inputs = kins_train, 
        outputs_tensor = outs_train,
        epochs = CONFIG['epochs'], 
        batch = CONFIG['batch_size']
        )
    
    cffs_pred = model_class.model(kins_train, training=False).numpy()

    cffs_fit = cffs_pred[0]
    phi_fit = tf.convert_to_tensor(df_outputs['phi'], dtype=tf.float32)
    dsig_fit = kinematics.calculate_cross_section(phi_fit, cffs_pred)

    result_csv = pd.DataFrame({'set': set_id, 'replica': replica_id, 'phi': np.rad2deg(np.pi-phi_fit), 
                               'dsig_sample': sample_dsig, 'dsig_fit': dsig_fit,
                               'ReH_pred': cffs_fit[0], 'ReHt_pred': cffs_fit[1], 
                               'ReE_pred': cffs_fit[2], 'DVCS_pred': cffs_fit[3]})
    
    history_csv = pd.DataFrame(history.history)
    history_csv.insert(0, "set", set_id)
    history_csv.insert(1, "replica", replica_id)

    del model_class

    return history_csv, result_csv

def worker(args):
    
    CONFIG, set_id, replica_id = args
    
    if replica_id==1:
        os.system("clear")
        print(f'>> Set {set_id} Progress:\n')

    filename = 'pseudo_KM15_BKM10_Jlab_all_t2_8pars.csv'

    df = pd.read_csv(filename, usecols=[0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18], skiprows=1,
                    names=['set', 'k', 'Q2', 'x', 't', 'phi', 'dsig_exp', 'dsig_err',
                            'ReH', 'ReE', 'ReHt', 'ReEt', 'ImH', 'ImE', 'ImHt', 'ImEt', 'DVCS'])

    # only choose columns for which experimental data exists.
    df = df[df['dsig_exp'] > 0]

    # angle in trento convention
    df['phi'] = np.pi-np.deg2rad(df['phi'])

    df = df[df['set'] == set_id]

    df_cffs = df[['ReH', 'ReHt', 'ReE', 'DVCS']].copy()

    df_kins = df[['Q2', 'x', 't']].copy()

    df_dsig = df[['phi', 'dsig_exp', 'dsig_err']].copy()

    weights = 1/(df_dsig['dsig_err'])
    weights /= np.sum(weights)

    df_dsig['weights'] = weights

    k = df['k'].values[0]
    Q2 = df['Q2'].values[0]
    xb = df['x'].values[0]
    t = df['t'].values[0]

    kinematics = BKM10(k, Q2, xb, t)

    input_pipeline = make_pipeline(
        MinMaxScaler()
    )

    df_kins = input_pipeline.fit_transform(df_kins)

    history, result = run_replica(CONFIG, set_id, replica_id, kinematics, df_kins, df_dsig)

    gc.collect()
    
    return history, result

def run_set(CONFIG, folder_name, set_id, num_replicas, threads):
    
    ctx = mp.get_context("spawn")  # ensures new Python interpreters

    args_list = [(CONFIG, set_id, j+1) for j in range(num_replicas)]

    results = []
    with ProcessPoolExecutor(max_workers=threads, mp_context=ctx) as ex:
        futures = [ex.submit(worker, args) for args in args_list]
        for f in tqdm(as_completed(futures), total=len(futures)):
            results.append(f.result())

    all_histories, all_results = zip(*results)

    result_writer = pq.ParquetWriter(
        f'data/{folder_name}/sample/set_{set_id}.parquet',
        schema=pa.Table.from_pandas(all_results[0]).schema
    )

    history_writer = pq.ParquetWriter(
        f'data/{folder_name}/history/set_{set_id}.parquet',
        schema=pa.Table.from_pandas(all_histories[0]).schema
    )

    for i in range(len(all_results)):
        result_writer.write_table(pa.Table.from_pandas(all_results[i]))
        history_writer.write_table(pa.Table.from_pandas(all_histories[i]))
    result_writer.close()
    history_writer.close()