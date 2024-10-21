############################################################################
###############  Written by Ishara Fernando, Ani Venkatapuram   ############
##############  Revised Date: 10/07/2024    ################################
##### Rivanna usage: This code is to generate replica models and evaluation#      
######### with only relu6 activation function ##############################
############################################################################
############################################################################
### This can be used to run local fits individually,         ###############
### or in parallel on rivanna                                ###############
### Include following lines to submit parallel jobs on rivanna  ############
###  #SBATCH --array=0-196 (or the kinematic sets you want)     ############
### python (file name) $SLURM_ARRAY_TASK_ID                     ############
###                   IMPORTANT:                                ############
###   Change scratch directory to your own computing id         ############
############################################################################

import numpy as np
import pandas as pd
import tensorflow as tf
from BHDVCS_tf_modified import *
import matplotlib.pyplot as plt
import os
import sys
from scipy.stats import norm


def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")
        


data_file = 'Basic_Model_pseudo_data_for_Jlab_kinematics.csv'
#df = pd.read_csv(data_file, dtype=np.float64)
df = pd.read_csv(data_file)
df = df.rename(columns={"sigmaF": "errF"})

## Here provide the name of the folder that you want to save your models, loss_plots, and replica_data ##
test_folder_name = 'Test_Arc_02'
scratch_loction = '/scratch/qzf7nj/DNN_CFFs_test_error_band/Tests/'
scratch_path_folder = str(scratch_loction) + str(test_folder_name)
create_folders(str(scratch_path_folder))
scratch_path = scratch_path_folder+'/'
#scratch_path = '/scratch/cee9hc/DNN_CFFs/Tests/Test_Arc_00/'

#### User's inputs ####
#Learning_Rate = 0.0001
Learning_Rate = 0.1
EPOCHS = 10
BATCH = 10
EarlyStop_patience = 1000
modify_LR_patience = 400
modify_LR_factor = 0.9

NUM_REPLICAS = 5

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
    x1 = tf.keras.layers.Dense(480, activation="relu6", kernel_initializer=initializer)(kinematics)
    x2 = tf.keras.layers.Dense(320, activation="relu6", kernel_initializer=initializer)(x1)
    x3 = tf.keras.layers.Dense(240, activation="relu6", kernel_initializer=initializer)(x2)
    x4 = tf.keras.layers.Dense(120, activation="relu6", kernel_initializer=initializer)(x3)
    x5 = tf.keras.layers.Dense(32, activation="relu6", kernel_initializer=initializer)(x4)
    outputs = tf.keras.layers.Dense(4, activation="linear", kernel_initializer=initializer, name='cff_output_layer')(x5)
    #### QQ, x_b, t, phi, k, cffs ####
    total_FInputs = tf.keras.layers.concatenate([inputs, outputs], axis=1)
    TotalF = TotalFLayer(name='TotalFLayer')(total_FInputs) # get rid of f1 and f2
    tfModel = tf.keras.Model(inputs=inputs, outputs = TotalF, name="tfmodel")
    tfModel.compile(
        optimizer = tf.keras.optimizers.Adam(Learning_Rate),
        loss = tf.keras.losses.MeanSquaredError()
    )
    return tfModel


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

def run_replica(kinset,i,xdf):
    replica_number = i
    tempdf = GenerateReplicaData(xdf)
    tempdf.to_csv(str(scratch_path) + 'ReplicaData_Kin_Set_'+str(j) + '/rep_data' + str(replica_number) + '.csv')

    trainKin, testKin, trainY, testY, trainYerr, testYerr = split_data(tempdf[['QQ', 'x_b', 't', 'phi_x', 'k']],
                                                                       tempdf['F'], tempdf['errF'], split=0.1)

    tfModel = DNNmodel()
    history = tfModel.fit(trainKin, trainY, validation_data=(testKin, testY), epochs=EPOCHS, callbacks=[modify_LR],
                          batch_size=BATCH, verbose=2)
    
    tfModel.save(str(scratch_path)+'DNNmodels_Kin_Set_'+ str(kinset)+ '/' + 'model' + str(replica_number) + '.h5', save_format='h5')

    tempX = df[['QQ', 'x_b', 't', 'phi_x', 'k']]
    PredictedFs = np.array(tf.keras.backend.function(tfModel.get_layer(name='input_layer').input, 
                                                      tfModel.get_layer(name='TotalFLayer').output)(tempX))
    
    # Store the predictions for later plotting
    if 'replica_predictions' not in globals():
        global replica_predictions
        replica_predictions = []
    
    replica_predictions.append(PredictedFs.flatten())
    
    # Create subplots for loss plots
    plt.figure(1, figsize=(12, 5))
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Val. loss')
    plt.title('Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(str(scratch_path)+'Losses_Plots_Kin_Set_'+str(kinset)+'/' + 'loss_plots_' + str(replica_number) + '.pdf')
    plt.close()

def plot_f_vs_phi(kin_df, replica_predictions, kinset):
    plt.figure(figsize=(10, 6))
    
    # Plot the original data points
    plt.errorbar(kin_df['phi_x'], kin_df['F'], kin_df['errF'], fmt='o', label="Data", color='blue', markersize=5)

    # Plot each replica's predictions
    for i, replica in enumerate(replica_predictions):
        plt.plot(kin_df['phi_x'], replica, label=f'Replica {i+1}', linestyle='--', alpha=0.6)

    plt.title(f'F vs Phi for Kinematic Set {kinset}')
    plt.xlabel(r'$\phi_x$')
    plt.ylabel('F')
    plt.legend(loc='best', fontsize='small')
    
    output_file = str(scratch_path) + f'F_vs_Phi_Kinematic_Set_{kinset}.pdf'
    plt.savefig(output_file)
    plt.close()



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
    print(f"Predicted F shape: {f_values.shape}")
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

# Take the real F values and phi_x values for the current kinematic set
real_F_values = kin_df['F'].values
phi_x_values = kin_df['phi_x'].values

# Get prediction inputs
prediction_inputs = kin_df[['QQ', 'x_b', 't', 'phi_x', 'k']].to_numpy()

# Get true CFF values
set_data = kin_df[['ReH', 'ReE', 'ReHt', 'dvcs']]
true_values = set_data.iloc[0].values

# Lists to store the predicted values
f_predictions = []
cff_predictions = []

# Predict CFFs and F values for each model
for model_id in models:
    tempFLayer, tempCFFsLayer = load_FLayer_and_cffLayer(model_id)
    cffs, f_values = predict_cffs_and_f(tempCFFsLayer, tempFLayer, prediction_inputs)
    f_predictions.append(f_values)  
    cff_predictions.append(cffs)


# # Function to append residual data to a CSV
# def save_cff_data_to_csv(cff_predictions, true_values, cff_labels, set_number, output_csv='CFFs_Comparison_AllSets.csv'):
#     """
#     Appends data (True, Predicted, Residual, Std Dev) for each CFF to a CSV file for all sets.
#     If the CSV already exists, the new data is appended; otherwise, a new file is created.
#     """
#     row_data = {'Set_Number': set_number}  # Set number as the first column
    
#     # Initialize lists to store residuals and std deviations for each CFF
#     residuals = []
#     std_devs = []
    
#     for i, cff_label in enumerate(cff_labels):
#         data = np.array(cff_predictions)[:, :, i].T.flatten()
#         mean_value = np.mean(data)
#         std_deviation = np.std(data)
#         residual = true_values[i] - mean_value
        
#         # Append residual and standard deviation for the current CFF
#         residuals.append(residual)
#         std_devs.append(std_deviation)
        
#         # Append the data to row_data for CSV
#         row_data[f'{cff_label}_True'] = true_values[i]
#         row_data[f'{cff_label}_Predicted'] = mean_value
#         row_data[f'{cff_label}_Residual'] = residual
#         row_data[f'{cff_label}_Std_Dev'] = std_deviation
    
#     # Convert the row to a DataFrame
#     df = pd.DataFrame([row_data])
    
#     # Append to CSV or create a new one
#     if not os.path.isfile(output_csv):
#         df.to_csv(output_csv, index=False, mode='w')
#         print(f"CSV file created: {output_csv}")
#     else:
#         df.to_csv(output_csv, index=False, mode='a', header=False)
#         print(f"Data appended to CSV: {output_csv}")
    
#     return residuals, std_devs


# def append_residuals_to_plot(cff_label, residuals, std_devs, set_number, temp_data):
#     """
#     Append residuals and std_devs for the current CFF to the temp_data structure,
#     which will later be used to create violin plots for each CFF.
#     """
#     if cff_label not in temp_data:
#         temp_data[cff_label] = {'kinematic_sets': [], 'residuals': [], 'std_devs': []}
    
#     # Append the entire list of residuals and std_devs
#     temp_data[cff_label]['kinematic_sets'].append(set_number)
#     temp_data[cff_label]['residuals'].append(residuals)  # List of residuals
#     temp_data[cff_label]['std_devs'].append(std_devs)    # List of std_devs


# def generate_violin_plots_for_cffs(temp_data, cff_labels, output_dir='ViolinPlots'):
#     """
#     Generates and saves violin plots for each CFF using the data stored in temp_data.
    
#     Parameters:
#     - temp_data: A dictionary that holds residuals and std_devs for each CFF across kinematic sets.
#     - cff_labels: List of CFF labels (e.g., ['ReH', 'ReE', 'ReHt', 'dvcs']).
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     for cff_label, data in temp_data.items():
#         kinematic_sets = data['kinematic_sets']
#         residuals_list = data['residuals']
#         std_devs_list = data['std_devs']

#         # Ensure kinematic_sets are numeric
#         kinematic_sets = [int(k) for k in kinematic_sets]  # Convert to integer if needed

#         # Prepare the data for violin plot
#         plot_data = []
#         for residual, std_dev in zip(residuals_list, std_devs_list):
#             generated_data = np.random.normal(loc=residual, scale=std_dev, size=1000)
#             plot_data.append(generated_data)

#         # Create the violin plot
#         plt.figure(figsize=(10, 6))
#         parts = plt.violinplot(plot_data, positions=kinematic_sets, widths=0.7, showmeans=False, showmedians=True, bw_method=0.5)

#         for pc in parts['bodies']:
#             pc.set_facecolor('gray')
#             pc.set_edgecolor('black')
#             pc.set_alpha(0.7)

#         plt.axhline(y=0, color='black', linestyle='--', linewidth=1)  # Residuals centered around 0
#         plt.xticks(kinematic_sets)
#         plt.xlabel('Kinematic Set Number')
#         plt.ylabel('Residual')
#         plt.title(f'Violin Plot of {cff_label} Residuals Across Kinematic Sets')

#         # Save the plot
#         plot_path = os.path.join(output_dir, f'{cff_label}_Residuals_ViolinPlot.png')
#         plt.tight_layout()
#         plt.savefig(plot_path)
#         plt.close()
#         print(f"Violin plot saved: {plot_path}")




# cff_labels = ['ReH', 'ReE', 'ReHt', 'dvcs']
# # Integration into your workflow
# temp_data = {cff: {'kinematic_sets': [], 'residuals': [], 'std_devs': []} for cff in cff_labels}  # Temporary storage for residuals and standard deviations


# plt.figure(figsize=(15, 10))

# for i, cff_label in enumerate(cff_labels):
#     plt.subplot(2, 2, i+1)
#     data = np.array(cff_predictions)[:, :, i].T.flatten()
#     plt.hist(data, bins=20, edgecolor='black', alpha=0.7, color='lightblue')

#     mean_value = np.mean(data)
#     std_deviation = np.std(data)

#     # Plot vertical lines for true value, mean, and bounds for 1-sigma
#     plt.axvline(x=true_values[i], color='red', linestyle='--', label='True Value')
#     plt.axvline(x=mean_value, color='blue', linestyle='--', label='Mean')
#     plt.axvline(x=mean_value - std_deviation, color='green', linestyle='--', label='1-sigma')
#     plt.axvline(x=mean_value + std_deviation, color='green', linestyle='--')

#     plt.title(f'Set {set_number}: {cff_label} Histogram\n Mean: {mean_value:.4f}, Std Dev: {std_deviation:.4f}')
#     plt.xlabel(cff_label)
#     plt.ylabel('Frequency')
#     plt.legend()

# # Save the CFF histograms as a PDF file
# output_pdf_path_combined = 'Comparison_Plots/' + f'CFFs_CombinedPlots_kinematic_set_{set_number}.pdf'
# plt.tight_layout()
# plt.savefig(output_pdf_path_combined)
# plt.close()

# # Use the current set number (e.g., j)
# set_number = j

# # Save CFF data to a CSV for the current set and collect residuals/std_devs for plotting
# residuals, std_devs = save_cff_data_to_csv(cff_predictions, true_values, cff_labels, set_number, output_csv='CFFs_Comparison_AllSets.csv')

# # Append the data for each CFF to the temp_data dictionary
# for i, cff_label in enumerate(cff_labels):
#     append_residuals_to_plot(cff_label, [residuals[i]], [std_devs[i]], set_number, temp_data)  # Pass as list


# # Once all sets are processed, generate violin plots
# generate_violin_plots_for_cffs(temp_data, cff_labels)



# # # Create subplots in a single figure for CFF histograms with vertical lines
# # plt.figure(figsize=(15, 10))
# # cff_labels = ['ReH', 'ReE', 'ReHt', 'dvcs']
# # for i, cff_label in enumerate(cff_labels):
# #     plt.subplot(2, 2, i+1)
# #     data = np.array(cff_predictions)[:, :, i].T.flatten()
# #     plt.hist(data, bins=20, edgecolor='black', alpha=0.7, color='lightblue')

# #     mean_value = np.mean(data)
# #     std_deviation = np.std(data)

# #     # Plot vertical lines for true value, mean, and bounds for 1-sigma
# #     plt.axvline(x=true_values[i], color='red', linestyle='--', label='True Value')
# #     plt.axvline(x=mean_value, color='blue', linestyle='--', label='Mean')
# #     plt.axvline(x=mean_value - std_deviation, color='green', linestyle='--', label='1-sigma')
# #     plt.axvline(x=mean_value + std_deviation, color='green', linestyle='--')

# #     # Fit a Gaussian curve with the correct x-axis limits
# #     xmin, xmax = plt.xlim()
# #     x = np.linspace(xmin, xmax, 100)
# #     p = norm.pdf(x, mean_value, std_deviation)
# #     plt.plot(x, p * len(data) * (xmax - xmin) / 20, 'k', linewidth=2)

# #     plt.title(f'Set {set_number}: {cff_label} Histogram (from Local)\n Mean: {mean_value:.4f}, Std Dev: {std_deviation:.4f}')
# #     plt.xlabel(cff_label)
# #     plt.ylabel('Frequency')
# #     plt.legend()

# # # Save the CFF histograms as a PDF file
# # output_pdf_path_combined = 'Comparison_Plots/'+'CFFs_CombinedPlots_subplot_kinematic_set_'+ str(j) +'.pdf'
# # plt.tight_layout()
# # plt.savefig(output_pdf_path_combined)
# # plt.close()

# # set_number = j  # Use the current set number

# # csv_data = append_cff_data_to_csv(csv_data, cff_predictions, true_values, cff_labels, set_number)
# # csv_file_path = 'CFFs_Comparison_AllSets.csv'
# # csv_data.to_csv(csv_file_path, index=False)
# # print(f"CSV file saved: {csv_file_path}")

# # save the chi-square error file
# chi_square_file = 'Comparison_Plots/chi_square_errors.txt'

# # Create or clear the chi-square file if this is the first kinematic set
# if set_number == 1 and os.path.exists(chi_square_file):
#     with open(chi_square_file, 'w') as file:
#         file.write("Kinematic Set\tChi-Square Error\n")  # Add headers if needed

# # Now create a plot for F vs phi_x, with real F points, predicted F values, error bands, and error bars
# plt.figure(figsize=(10, 6))

# # Initialize lists to store the mean and standard deviation for each phi_x
# mean_f_predictions = []
# std_f_predictions = []

# # Loop over each phi_x value
# for i in range(len(phi_x_values)):
#     # Extract the predicted F values for the i-th phi_x from all replicas
#     f_values_at_phi_x = [f_pred[i] for f_pred in f_predictions]

#     # Calculate the mean and standard deviation for this phi_x
#     mean_f_predictions.append(np.mean(f_values_at_phi_x))
#     std_f_predictions.append(np.std(f_values_at_phi_x))

# # Convert the mean and std lists to arrays for easier plotting
# mean_f_predictions = np.array(mean_f_predictions)
# std_f_predictions = np.array(std_f_predictions)

# # Calculate chi-square error for the kinematic set
# chi_square_error = np.sum(((real_F_values - mean_f_predictions) / std_f_predictions) ** 2)

# # Save the chi-square error to the file
# with open(chi_square_file, 'a') as file:
#     file.write(f"{set_number}\t{chi_square_error:.4f}\n")

# # Plot real F values as red scatter points
# plt.scatter(phi_x_values, real_F_values, color='red', label='Real F', zorder=5)

# # Add red error bars to the real F values (±1 standard deviation)
# plt.errorbar(phi_x_values, real_F_values, yerr=std_f_predictions, fmt='o', color='red', ecolor='red', capsize=5, label='Real F Error', zorder=6)

# # Plot the mean predicted F values as a blue line
# plt.plot(phi_x_values, mean_f_predictions, color='blue', label='Mean F Prediction')

# # Plot the error bands (mean ± 1 standard deviation)
# plt.fill_between(phi_x_values, mean_f_predictions - std_f_predictions, mean_f_predictions + std_f_predictions, 
#                  color='blue', alpha=0.2, label='±1 Std Dev')

# plt.xlabel('phi_x')
# plt.ylabel('F')
# plt.title(f'F vs phi_x with Error Bars and Error Bands for Kinematic Set {set_number}')
# plt.grid(True)
# plt.legend()

# # Save the F vs phi_x plot with error bars and error bands as a PNG file
# output_png_path = f'Comparison_Plots/F_vs_phi_x_with_error_bars_and_bands_Kinematic_Set_{set_number}.png'
# plt.savefig(output_png_path)
# plt.close()

# print(f"Kinematic Set {set_number}: Chi-Square Error = {chi_square_error:.4f}")

# Function to append residual data to a CSV
def save_cff_data_to_csv(cff_predictions, true_values, cff_labels, set_number, output_csv='CFFs_Comparison_AllSets.csv'):
    """
    Appends data (True, Predicted, Residual, Std Dev) for each CFF to a CSV file for all sets.
    If the CSV already exists, the new data is appended; otherwise, a new file is created.
    """
    row_data = {'Set_Number': set_number}  # Set number as the first column
    
    # Initialize lists to store residuals and std deviations for each CFF
    residuals = []
    std_devs = []
    
    for i, cff_label in enumerate(cff_labels):
        data = np.array(cff_predictions)[:, :, i].T.flatten()
        mean_value = np.mean(data)
        std_deviation = np.std(data)
        residual = true_values[i] - mean_value
        
        # Append residual and standard deviation for the current CFF
        residuals.append(residual)
        std_devs.append(std_deviation)
        
        # Append the data to row_data for CSV
        row_data[f'{cff_label}_True'] = true_values[i]
        row_data[f'{cff_label}_Predicted'] = mean_value
        row_data[f'{cff_label}_Residual'] = residual
        row_data[f'{cff_label}_Std_Dev'] = std_deviation
    
    # Convert the row to a DataFrame
    df = pd.DataFrame([row_data])
    
    # Append to CSV or create a new one
    if not os.path.isfile(output_csv):
        df.to_csv(output_csv, index=False, mode='w')
        print(f"CSV file created: {output_csv}")
    else:
        df.to_csv(output_csv, index=False, mode='a', header=False)
        print(f"Data appended to CSV: {output_csv}")
    
    return residuals, std_devs

def generate_violin_plots_from_csv(output_csv='CFFs_Comparison_AllSets.csv', cff_labels=['ReH', 'ReE', 'ReHt', 'dvcs'], output_dir='ViolinPlots'):
    """
    Generates and saves violin plots for each CFF using the data from the CSV file.
    
    Parameters:
    - output_csv: Path to the CSV file containing residuals and std deviations.
    - cff_labels: List of CFF labels (e.g., ['ReH', 'ReE', 'ReHt', 'dvcs']).
    - output_dir: Directory where the violin plots will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the CSV file
    df = pd.read_csv(output_csv)
    
    for cff_label in cff_labels:
        # Extract kinematic sets, residuals, and std deviations for the current CFF
        kinematic_sets = df['Set_Number'].unique()
        kinematic_sets = sorted(kinematic_sets)  # Ensure ordered
        
        plot_data = []
        valid_sets = []
        
        for set_num in kinematic_sets:
            residual = df.loc[df['Set_Number'] == set_num, f'{cff_label}_Residual']
            std_dev = df.loc[df['Set_Number'] == set_num, f'{cff_label}_Std_Dev']
            
            if not residual.empty and not std_dev.empty:
                residual_val = residual.values[0]
                std_dev_val = std_dev.values[0]
                
                # Generate data for violin plot
                generated_data = np.random.normal(loc=residual_val, scale=std_dev_val, size=1000)
                plot_data.append(generated_data)
                valid_sets.append(set_num)
            else:
                print(f"Warning: Missing data for CFF '{cff_label}' in Set {set_num}. Skipping this set.")
        
        if not plot_data:
            print(f"No data available for CFF '{cff_label}'. Skipping violin plot generation.")
            continue
    
        # Create the violin plot
        plt.figure(figsize=(10, 6))
        parts = plt.violinplot(plot_data, positions=valid_sets, widths=0.7, showmeans=False, showmedians=True, bw_method=0.5)
    
        # Update color to gray
        for pc in parts['bodies']:
            pc.set_facecolor('gray')
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
    
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)  # Residuals centered around 0
        plt.xticks(valid_sets)
        plt.xlabel('Kinematic Set Number')
        plt.ylabel('Residual')
        plt.title(f'Violin Plot of {cff_label} Residuals Across Kinematic Sets')
        plt.grid(axis='y')
    
        # Save the plot
        plot_path = os.path.join(output_dir, f'{cff_label}_Residuals_ViolinPlot.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Violin plot saved: {plot_path}")

# Function to generate other plots (e.g., Histograms, F vs phi_x)
def generate_other_plots(set_number, cff_predictions, true_values, cff_labels, 
                         phi_x_values, real_F_values, f_predictions, output_dir='Comparison_Plots'):
    """
    Generates histograms for CFFs and F vs phi_x plots.
    
    Parameters:
    - set_number: Current kinematic set number.
    - cff_predictions: Predicted CFF values.
    - true_values: True CFF values.
    - cff_labels: List of CFF labels.
    - phi_x_values: Values of phi_x.
    - real_F_values: Real F values corresponding to phi_x.
    - f_predictions: Predicted F values from replicas.
    - output_dir: Directory where the plots will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(15, 10))
    
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
    
        plt.title(f'Set {set_number}: {cff_label} Histogram\n Mean: {mean_value:.4f}, Std Dev: {std_deviation:.4f}')
        plt.xlabel(cff_label)
        plt.ylabel('Frequency')
        plt.legend()
    
    # Save the CFF histograms as a PDF file
    output_pdf_path_combined = os.path.join(output_dir, f'CFFs_CombinedPlots_kinematic_set_{set_number}.pdf')
    plt.tight_layout()
    plt.savefig(output_pdf_path_combined)
    plt.close()
    
    # Save the chi-square error file
    chi_square_file = os.path.join(output_dir, 'chi_square_errors.txt')
    
    # Create or clear the chi-square file if this is the first kinematic set
    if set_number == 1 and os.path.exists(chi_square_file):
        with open(chi_square_file, 'w') as file:
            file.write("Kinematic Set\tChi-Square Error\n")  # Add headers if needed
    
    # Now create a plot for F vs phi_x, with real F points, predicted F values, error bands, and error bars
    plt.figure(figsize=(10, 6))
    
    # Initialize lists to store the mean and standard deviation for each phi_x
    mean_f_predictions = []
    std_f_predictions = []
    
    # Loop over each phi_x value
    for i in range(len(phi_x_values)):
        # Extract the predicted F values for the i-th phi_x from all replicas
        f_values_at_phi_x = [f_pred[i] for f_pred in f_predictions]
    
        # Calculate the mean and standard deviation for this phi_x
        mean_f_predictions.append(np.mean(f_values_at_phi_x))
        std_f_predictions.append(np.std(f_values_at_phi_x))
    
    # Convert the mean and std lists to arrays for easier plotting
    mean_f_predictions = np.array(mean_f_predictions)
    std_f_predictions = np.array(std_f_predictions)
    
    # Calculate chi-square error for the kinematic set
    chi_square_error = np.sum(((real_F_values - mean_f_predictions) / std_f_predictions) ** 2)
    
    # Save the chi-square error to the file
    with open(chi_square_file, 'a') as file:
        file.write(f"{set_number}\t{chi_square_error:.4f}\n")
    
    # Plot real F values as red scatter points
    plt.scatter(phi_x_values, real_F_values, color='red', label='Real F', zorder=5)
    
    # Add red error bars to the real F values (±1 standard deviation)
    plt.errorbar(phi_x_values, real_F_values, yerr=std_f_predictions, fmt='o', color='red', 
                 ecolor='red', capsize=5, label='Real F Error', zorder=6)
    
    # Plot the mean predicted F values as a blue line
    plt.plot(phi_x_values, mean_f_predictions, color='blue', label='Mean F Prediction')
    
    # Plot the error bands (mean ± 1 standard deviation)
    plt.fill_between(phi_x_values, mean_f_predictions - std_f_predictions, mean_f_predictions + std_f_predictions, 
                     color='blue', alpha=0.2, label='±1 Std Dev')
    
    plt.xlabel('phi_x')
    plt.ylabel('F')
    plt.title(f'F vs phi_x with Error Bars and Error Bands for Kinematic Set {set_number}')
    plt.grid(True)
    plt.legend()
    
    # Save the F vs phi_x plot with error bars and error bands as a PNG file
    output_png_path = os.path.join(output_dir, f'F_vs_phi_x_with_error_bars_and_bands_Kinematic_Set_{set_number}.png')
    plt.savefig(output_png_path)
    plt.close()
    
    print(f"Kinematic Set {set_number}: Chi-Square Error = {chi_square_error:.4f}")

# Example integration into your workflow
def main_workflow(cff_predictions_all_sets, true_values, cff_labels, phi_x_values, real_F_values, f_predictions_all_sets, output_csv='CFFs_Comparison_AllSets.csv'):
    """
    Main workflow to process multiple kinematic sets, save residuals to CSV, and generate plots.
    
    Parameters:
    - cff_predictions_all_sets: List or iterable containing CFF predictions for each set.
    - true_values: List of true CFF values.
    - cff_labels: List of CFF labels.
    - phi_x_values: List or array of phi_x values.
    - real_F_values: Array of real F values corresponding to phi_x.
    - f_predictions_all_sets: List or iterable containing F predictions for each set.
    - output_csv: Path to the CSV file for residuals and std deviations.
    """
    output_dir = 'Comparison_Plots'
    violin_output_dir = 'ViolinPlots'
    
    # Iterate over all kinematic sets
    for j, (cff_predictions, f_predictions, set_number) in enumerate(zip(cff_predictions_all_sets, f_predictions_all_sets, range(1, len(cff_predictions_all_sets)+1)), start=1):
        # Generate histograms and F vs phi_x plots
        generate_other_plots(set_number, cff_predictions, true_values, cff_labels, 
                             phi_x_values, real_F_values, f_predictions, output_dir=output_dir)
        
        # Save CFF data to CSV for the current set
        residuals, std_devs = save_cff_data_to_csv(cff_predictions, true_values, cff_labels, set_number, output_csv=output_csv)
        
        # Note: We are no longer using temp_data for violin plots
        # Violin plots will be generated after all sets are processed
    
    generate_violin_plots_from_csv(output_csv='CFFs_Comparison_AllSets.csv', cff_labels=['ReH', 'ReE', 'ReHt', 'dvcs'], output_dir='ViolinPlots')



if __name__ == "__main__":
    main_workflow()



