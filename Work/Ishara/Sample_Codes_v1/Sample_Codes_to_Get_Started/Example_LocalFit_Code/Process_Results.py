###################################
##  Written by Ishara Fernando   ##
##  Revised Date: 01/16/2024     ##
###################################

import numpy as np
import pandas as pd
import tensorflow as tf
from BHDVCS_tf_modified import *
import matplotlib.pyplot as plt
import plotly.express as px
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import plotly.graph_objects as go
#from plotly.subplots import make_subplots
import os
import sys


def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")
        
        
def load_FLayer_and_cffLayer(model):
    LayerF = tf.keras.models.load_model(model, custom_objects={'TotalFLayer': TotalFLayer})

    LayerCFFs = tf.keras.Model(inputs=LayerF.input,
                                              outputs=LayerF.get_layer('cff_output_layer').output)
    return LayerF, LayerCFFs


def predict_cffs_and_f(LayerCFFs, LayerF, inputs):
    cffs = LayerCFFs.predict(inputs)
    f_values = LayerF.predict(inputs)
    return cffs, f_values


model_folder = 'DNNmodels'
models = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if f.endswith('.h5')]


grid_file = 'Sample_Local_PseudoData_from_the_Basic_Model_for_JLab_Kinematics.csv'
grid_df = pd.read_csv(grid_file, dtype=np.float64)


models_folder = 'DNNmodels'
models = [os.path.join(models_folder, f) for f in os.listdir(models_folder) if f.endswith('.h5')]

f_predictions = []
cff_predictions = []

prediction_inputs = grid_df[['QQ', 'x_b', 't', 'phi_x', 'k']].to_numpy()

for modelid in models:
    tempFLayer, tempCFFsLayer = load_FLayer_and_cffLayer(modelid)
    cffs, f_values = predict_cffs_and_f(tempCFFsLayer, tempFLayer, prediction_inputs)
    f_predictions.append(f_values)
    cff_predictions.append(cffs)
    

def process_columns_with_acc(folder_path):
    # Get a list of all .csv files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # Initialize empty dataframes to store the columns from each file
    columns_df = {col: pd.DataFrame() for col in ['F', 'ReH', 'ReE', 'ReHt', 'dvcs', 'AbsRes_ReH', 'AbsRes_ReE', 'AbsRes_ReHt', 'AbsRes_dvcs']}

    # Iterate through each .csv file
    for file in csv_files:
        # Read the current file into a dataframe
        current_df = pd.read_csv(os.path.join(folder_path, file))

        # Iterate through the desired columns
        for col in columns_df.keys():
            # Check if the column exists in the current file
            if col in current_df.columns:
                # Extract the column and add it to the dataframe
                columns_df[col][file] = current_df[col]

    # Check if there are any columns in columns_df
    if not all(df.empty for df in columns_df.values()):
        # Calculate the mean and standard deviation for each column
        mean_columns = {col: df.mean(axis=1) for col, df in columns_df.items()}
        std_columns = {col: df.std(axis=1) for col, df in columns_df.items()}

        # Choose any file to get non-'F', 'ReE', 'ReHt', 'dvcs', 'AbsRes_ReH', 'AbsRes_ReE', 'AbsRes_ReHt', 'AbsRes_dvcs' columns (assuming the structure is the same for all files)
        example_file = pd.read_csv(os.path.join(folder_path, csv_files[0]))
        non_columns = example_file[['k', 'QQ', 'x_b', 't', 'phi_x']]

        # Create the final dataframe with 'k', 'QQ', 'x_b', 't', 'phi_x', mean, and std columns for each desired column
        final_df = non_columns.copy()
        for col in columns_df.keys():
            final_df[f'{col}'] = mean_columns[col]
            final_df[f'std_{col}'] = std_columns[col]

        return final_df
    else:
        print(f"No data found for columns 'F', 'ReH', 'ReE', 'ReHt', 'dvcs', 'AbsRes_ReH', 'AbsRes_ReE', 'AbsRes_ReHt', 'AbsRes_dvcs' in any of the files.")
        return None

    

## Here we generate a single file with the average values from all replica models ###
replicatestdf = process_columns_with_acc('Replica_Results')
replicatestdf.to_csv('Residual_summmary_from_Replicas.csv')



def create_4D_scatter_plot_HTML(df, acc_column, acc_range, title, filename):
    filtered_df = df[(df[acc_column] > acc_range[0]) & (df[acc_column] <= acc_range[1])]
    fig = px.scatter_3d(filtered_df, x='x_b', y='QQ', z='t', color=acc_column, title=title, color_continuous_scale='viridis')
    fig.update_layout(scene=dict(xaxis_title='x_b', yaxis_title='QQ', zaxis_title='t'))
    fig.write_html(filename)
    
create_4D_scatter_plot_HTML(replicatestdf, 'AbsRes_ReH', (0, 1), 'ReH Accuracy', 'kinematics_ReH_Accuracy.html')

