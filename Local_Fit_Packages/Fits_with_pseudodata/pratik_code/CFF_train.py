
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
import numpy as np
import pandas as pd
import keras
from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from bkm10 import diff_cross
import gc
# class Scaler(tf.keras.layers.Layer):
#     def __init__(self, data):
#         super().__init__()
#         self.log1p = True
#         log_data = keras.ops.log1p(data)
#         data_max = tf.reduce_max(data).numpy()
#         if (data_max < 0.1):
#             log_data = keras.ops.log10(data)
#             self.log1p = False
#         self.min = tf.reduce_min(log_data)
#         self.max = tf.reduce_max(log_data)
#     def transform(self, data):
#         if self.log1p:
#             return (keras.ops.log1p(data)-self.min)/(self.max-self.min)
#         return (keras.ops.log10(data)-self.min)/(self.max-self.min)
class Scaler(tf.keras.layers.Layer):
    def __init__(self, data, k):
        super().__init__()
        self.k5p75=False
        if k==5.75:
            self.k5p75=True
    def transform(self, data):
        if self.k5p75:
            return keras.ops.log1p(data)
        return keras.ops.log10(data)
        # return (data)
# class Scaler(tf.keras.layers.Layer):
#     def __init__(self, data):
#         super().__init__()
#         log_data = keras.ops.log1p(data)
#         self.min = tf.reduce_min(log_data)
#         self.max = tf.reduce_max(log_data)
#     def transform(self, data):        
#         return (keras.ops.log1p(data)-self.min)/(self.max-self.min)
filename = 'pseudo_KM15_BKM10_Jlab_all_t2_8pars.csv'
set_num, replica = map(int, input().split())
df = pd.read_csv(filename, usecols=[0, 2, 3, 4, 5, 6, 7, 8, 10, 12, 11, 18], skiprows=1,
                 names=['set', 'k', 'Q2', 'x', 't', 'phi', 'dsig', 'err_dsig', 'ReH', 'ReE', 'ReHt', 'DVCS'])
df = df[df['set']==set_num]
# only choose columns for which experimental data exists.
df = df[df['dsig'] > 0]
phi_dsig = df['phi']
# angle in trento convention
df['phi'] = np.pi-np.deg2rad(df['phi'])
df_cffs = df[['ReH', 'ReHt', 'ReE', 'DVCS']].copy()
df_kins = df[['k', 'Q2', 'x', 't', 'phi']].copy()
df_dsig_with_err = df[['dsig', 'err_dsig']].copy()
# model dsig from KM15 CFFs
modelled_dsig = diff_cross(tf.convert_to_tensor(df_kins, dtype=tf.float32),
                           tf.convert_to_tensor(df_cffs, dtype=tf.float32))
err_dsig = df_dsig_with_err['err_dsig']
sample_dsig = np.random.normal(loc=modelled_dsig, scale=err_dsig)
while np.any(sample_dsig < 0):
    negative_mask = sample_dsig < 0
    sample_dsig[negative_mask] = np.random.normal(loc=modelled_dsig[negative_mask], scale=err_dsig[negative_mask])
df_dsig_with_err['dsig'] = sample_dsig
# shuffle the data
kins_train, dsig_with_err_train = shuffle(df_kins, df_dsig_with_err)
# scale the relevant kinematics which will only be Q2, x, t {cols 1,2,3}
# add these scaled variables as separate columns: this will be useful later to pass in the raw kinematics into the physics layer
input_pipeline = make_pipeline(
    MinMaxScaler()
)
# Q2_x_t_scaled = input_pipeline.fit_transform(
#     kins_train[['Q2', 'x', 't']].to_numpy())
Q2_x_t_scaled = kins_train[['Q2', 'x', 't']].to_numpy()
kins_train = np.column_stack([kins_train, Q2_x_t_scaled])
# convert to tensors
dsig_train, err_train = dsig_with_err_train['dsig'].values.reshape(
    -1, 1), dsig_with_err_train['err_dsig'].values.reshape(-1, 1)
kins_train = tf.convert_to_tensor(kins_train, dtype=tf.float32)
dsig_train = tf.convert_to_tensor(dsig_train, dtype=tf.float32)
output_pipeline = Scaler(dsig_train, df_kins['k'].values[0])
dsig_train = output_pipeline.transform(dsig_train)
def create_model(lr=1e-1):
    all_input_kins = keras.Input(shape=(8,))
    kins_raw = keras.layers.Lambda(
        lambda x: x[:, 0:5],
        output_shape=(5,)
    )(all_input_kins)
    Q2_x_t_scaled = keras.layers.Lambda(
        lambda x: x[:, 5:8],
        output_shape=(3,)
    )(all_input_kins)
    init = keras.initializers.GlorotUniform()
    hidden = keras.layers.Dense(100, kernel_initializer=init, activation='relu')(Q2_x_t_scaled)
    hidden = keras.layers.Dense(50,  kernel_initializer=init, activation='relu')(hidden)
    hidden = keras.layers.Dense(25,  kernel_initializer=init, activation='relu')(hidden)
    hidden = keras.layers.Dense(10,  kernel_initializer=init, activation='relu')(hidden)
    cff_123 = keras.layers.Dense(3, name='cff123')(hidden)
    DVCS = keras.layers.Dense(1, name='dvcs', activation='softplus')(hidden)
    predicted_cffs = keras.layers.Concatenate(
        name='predicted_cff_layer')([cff_123, DVCS])
    predicted_dsig = keras.layers.Lambda(
        lambda args: diff_cross(args[0], args[1]),
        output_shape=(1,)
    )([kins_raw, predicted_cffs])
    model = keras.Model(inputs=all_input_kins, outputs=output_pipeline.transform(predicted_dsig))
    model.compile(optimizer=keras.optimizers.Adam(lr),
                #   loss=keras.losses.Huber(delta=0.01))
                  loss='mae')
    return model
modifyLR = keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.8, patience=40, min_lr=1e-5, mode='auto')
EarlyStop = keras.callbacks.EarlyStopping(
    monitor='loss', patience=50, restore_best_weights=True)
replica_csv = pd.DataFrame({'phi':phi_dsig, 'model_dsig':modelled_dsig,
                            'err_dsig':err_dsig, 'sample_dsig': sample_dsig})
model = create_model()
weights = 1/dsig_with_err_train['err_dsig']
weights /= np.mean(weights)
history = model.fit(kins_train, dsig_train, sample_weight= weights,
                    epochs=300, batch_size=1, callbacks=[modifyLR, EarlyStop], verbose=0)
history_csv = pd.DataFrame(history.history)
# model predictions
N = 100
test_kins = df_kins.values[0]
test_kins = np.tile(test_kins, (N, 1))
set_tile = np.tile(set_num, (N, 1))
phi_min = df_kins['phi'].values[-1]
phi_max = df_kins['phi'].values[0]
test_phi = np.linspace(phi_min, phi_max, N)
test_kins[:, -1] = test_phi
kins_test_scaled = input_pipeline.fit_transform(test_kins[:, 1:4])
kins_test_input = np.column_stack([test_kins, kins_test_scaled])
cff_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.get_layer('predicted_cff_layer').output
)
predicted_cffs = cff_model(tf.convert_to_tensor(kins_test_input), training=False).numpy()
ReH_pred = predicted_cffs[:,0]
ReHt_pred = predicted_cffs[:,1]
ReE_pred = predicted_cffs[:,2]
DVCS_pred = predicted_cffs[:,3]
predicted_dsig = diff_cross(tf.convert_to_tensor(test_kins, dtype=tf.float32),
                    tf.convert_to_tensor(predicted_cffs, dtype=tf.float32))
result_csv = pd.DataFrame({'k':test_kins[:,0], 'Q2':test_kins[:,1],
                           'x':test_kins[:,2], 't':test_kins[:,3],
                            'phi':np.rad2deg(np.pi-test_kins[:,4]), 'dsig':predicted_dsig,
                            'ReH':ReH_pred, 'ReHt':ReHt_pred,
                            'ReE':ReE_pred, 'DVCS':DVCS_pred})
replica_csv.to_csv(f'./data/set_{set_num}/sample/sample{replica}.csv', index=False)
model.save(f'./data/set_{set_num}/model/model{replica}.keras')
history_csv.to_csv(f'./data/set_{set_num}/history/history{replica}.csv', index=False)
result_csv.to_csv(f'./data/set_{set_num}/core/replica{replica}.csv', index=False)
# clear memory
keras.backend.clear_session()
gc.collect()
