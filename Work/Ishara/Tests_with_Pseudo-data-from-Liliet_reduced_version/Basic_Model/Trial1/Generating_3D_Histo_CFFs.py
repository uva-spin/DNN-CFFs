###################################
##  Written by Ishara Fernando   ##
##  Revised Date: 01/25/2024     ##
###################################

import os
import pandas as pd
import plotly.graph_objects as go

# Load the data from CSV files
df_replicas = pd.read_csv('average_values_from_replicas.csv')
df_pseudo = pd.read_csv('pseudo_basic_BKM10_Jlab_all_t2.csv')

# Create a folder named 'CFFs_Histograms' if it doesn't exist
output_folder = 'CFFs_Histograms'
os.makedirs(output_folder, exist_ok=True)

# Create 3D scatter plot with mean values and error bars for ReH
fig_ReH = go.Figure()

# Add mean values in x and t space with error bars for replicas
fig_ReH.add_trace(go.Scatter3d(
    x=df_replicas['x_b'],
    y=df_replicas['t'],
    z=df_replicas['ReH'],
    mode='markers',
    marker=dict(
        size=4,
        color='green',
        opacity=0.5
    ),
    error_z=dict(
        type='data',
        array=df_replicas['sigmaReH'],
        thickness=2,
        visible=True
    ),
    name='DNN ReH Values (Replicas)',
))

# Add "True ReH" values without connecting them
fig_ReH.add_trace(go.Scatter3d(
    x=df_pseudo['x_b'],
    y=df_pseudo['t'],
    z=df_pseudo['ReH'],
    mode='markers',
    marker=dict(
        size=4,
        color='darkred',
        opacity=0.5
    ),
    name='True ReH Values (Pseudo)',
))

# Layout settings for ReH plot
fig_ReH.update_layout(
    scene=dict(
        xaxis=dict(title='x_b'),
        yaxis=dict(title='t', range=[-2, 0]),
        zaxis=dict(title='ReH'),
    ),
    title='3D Scatter Plot of DNN ReH Values (Replicas) and True ReH Values (Pseudo)',
)

# Save the ReH plot as an HTML file
output_path_ReH = os.path.join(output_folder, 'ReH_3D_plot_with_DNN_values_and_true_ReH_values.html')
fig_ReH.write_html(output_path_ReH)

# Create 3D scatter plot with mean values and error bars for ReE
fig_ReE = go.Figure()

# Add mean values in x and t space with error bars for replicas
fig_ReE.add_trace(go.Scatter3d(
    x=df_replicas['x_b'],
    y=df_replicas['t'],
    z=df_replicas['ReE'],
    mode='markers',
    marker=dict(
        size=4,
        color='green',
        opacity=0.5
    ),
    error_z=dict(
        type='data',
        array=df_replicas['sigmaReE'],
        thickness=2,
        visible=True
    ),
    name='DNN ReE Values (Replicas)',
))

# Add "True ReE" values without connecting them
fig_ReE.add_trace(go.Scatter3d(
    x=df_pseudo['x_b'],
    y=df_pseudo['t'],
    z=df_pseudo['ReE'],
    mode='markers',
    marker=dict(
        size=4,
        color='darkblue',
        opacity=0.5
    ),
    name='True ReE Values (Pseudo)',
))

# Layout settings for ReE plot
fig_ReE.update_layout(
    scene=dict(
        xaxis=dict(title='x_b'),
        yaxis=dict(title='t', range=[-2, 0]),
        zaxis=dict(title='ReE'),
    ),
    title='3D Scatter Plot of DNN ReE Values (Replicas) and True ReE Values (Pseudo)',
)

# Save the ReE plot as an HTML file
output_path_ReE = os.path.join(output_folder, 'ReE_3D_plot_with_DNN_values_and_true_ReE_values.html')
fig_ReE.write_html(output_path_ReE)

# Create 3D scatter plot with mean values and error bars for ReHt
fig_ReHt = go.Figure()

# Add mean values in x and t space with error bars for replicas
fig_ReHt.add_trace(go.Scatter3d(
    x=df_replicas['x_b'],
    y=df_replicas['t'],
    z=df_replicas['ReHt'],
    mode='markers',
    marker=dict(
        size=4,
        color='green',
        opacity=0.5
    ),
    error_z=dict(
        type='data',
        array=df_replicas['sigmaReHt'],
        thickness=2,
        visible=True
    ),
    name='DNN ReHt Values (Replicas)',
))

# Add "True ReHt" values without connecting them
fig_ReHt.add_trace(go.Scatter3d(
    x=df_pseudo['x_b'],
    y=df_pseudo['t'],
    z=df_pseudo['ReHt'],
    mode='markers',
    marker=dict(
        size=4,
        color='darkorange',
        opacity=0.5
    ),
    name='True ReHt Values (Pseudo)',
))

# Layout settings for ReHt plot
fig_ReHt.update_layout(
    scene=dict(
        xaxis=dict(title='x_b'),
        yaxis=dict(title='t', range=[-2, 0]),
        zaxis=dict(title='ReHt'),
    ),
    title='3D Scatter Plot of DNN ReHt Values (Replicas) and True ReHt Values (Pseudo)',
)

# Save the ReHt plot as an HTML file
output_path_ReHt = os.path.join(output_folder, 'ReHt_3D_plot_with_DNN_values_and_true_ReHt_values.html')
fig_ReHt.write_html(output_path_ReHt)

# Create 3D scatter plot with mean values and error bars for dvcs
fig_dvcs = go.Figure()

# Add mean values in x and t space with error bars for replicas
fig_dvcs.add_trace(go.Scatter3d(
    x=df_replicas['x_b'],
    y=df_replicas['t'],
    z=df_replicas['dvcs'],
    mode='markers',
    marker=dict(
        size=4,
        color='green',
        opacity=0.5
    ),
    error_z=dict(
        type='data',
        array=df_replicas['sigmadvcs'],
        thickness=2,
        visible=True
    ),
    name='DNN dvcs Values (Replicas)',
))

# Add "True dvcs" values without connecting them
fig_dvcs.add_trace(go.Scatter3d(
    x=df_pseudo['x_b'],
    y=df_pseudo['t'],
    z=df_pseudo['dvcs'],
    mode='markers',
    marker=dict(
        size=4,
        color='purple',
        opacity=0.5
    ),
    name='True dvcs Values (Pseudo)',
))

# Layout settings for dvcs plot
fig_dvcs.update_layout(
    scene=dict(
        xaxis=dict(title='x_b'),
        yaxis=dict(title='t', range=[-2, 0]),
        zaxis=dict(title='dvcs'),
    ),
    title='3D Scatter Plot of DNN dvcs Values (Replicas) and True dvcs Values (Pseudo)',
)

# Save the dvcs plot as an HTML file
output_path_dvcs = os.path.join(output_folder, 'dvcs_3D_plot_with_DNN_values_and_true_dvcs_values.html')
fig_dvcs.write_html(output_path_dvcs)

# Show all the plots
fig_ReH.show()
fig_ReE.show()
fig_ReHt.show()
fig_dvcs.show()

print(f'3D Plots with DNN values and true values saved in {output_folder} folder.')





