##########################################################################
###############  Written by Ishara Fernando                  #############
##############  Revised Date: 05/29/2024    ##############################
##### Rivanna usage: Run the following commands on your Rivanna terminal##
## source /home/lba9wf/miniconda3/etc/profile.d/conda.sh         #########
## conda activate env                                            #########
##########################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BHDVCS_tf_modified import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

fns = F1F2()
calc = F_calc()

########## INPUT from the User ###################

### Kinematics file ####
file_name = 'AllJlabData_from_Zulkaida_and_Liliet.csv'
########## The input parameters for a, b, c, d, e, f need to be provided below #####


kindf = pd.read_csv(file_name, dtype=np.float32).dropna(axis=0, how='all').dropna(axis=1, how='all')

### Generating function ###
def genf(x,t,a,b,c,d,e,f):
    temp = (a*(x**2) + b*x)*np.exp(c*(t**2)+d*t+e)+f
    return temp

# Basic 2 #

def ReHps(x,t):
    temp_a = -4.41
    temp_b = 1.68
    temp_c = -9.14
    temp_d = -3.57
    temp_e = 1.54
    temp_f = 2.07
    return genf(x,t,temp_a, temp_b, temp_c, temp_d, temp_e, temp_f)

def ReEps(x,t):
    temp_a = -1.04
    temp_b = 0.46
    temp_c = 0.6
    temp_d = 1.95
    temp_e = 2.72
    temp_f = -0.95
    return genf(x,t,temp_a, temp_b, temp_c, temp_d, temp_e, temp_f)

def ReHtps(x,t):
    temp_a = -1.86
    temp_b = 1.50
    temp_c = -0.29
    temp_d = -1.33
    temp_e = 0.46
    temp_f = -0.98
    return genf(x,t,temp_a, temp_b, temp_c, temp_d, temp_e, temp_f)

def DVCSps(x,t):
    temp_a = 0.52
    temp_b = -0.41
    temp_c = 0.05
    temp_d = -0.25
    temp_e = 0.55
    temp_f = 0.173
    return genf(x,t,temp_a, temp_b, temp_c, temp_d, temp_e, temp_f)


def GeneratePseudoData(df):
    pseudodata_df = {'set': [],
                     'k': [],
                     'QQ': [],
                     'x_b': [],
                     't': [],
                     'phi_x': [],
                     'F':[],
                     'sigmaF':[],                     
                     'ReH': [],
                     'ReE': [],
                     'ReHt': [],
                     'dvcs': []}
    for i in range(len(df)):
        row = df.loc[i]
        
        tempSet, tempQQ, tempxb, tempt, tempk, tempphi, varF = np.array([row['#Set'],row['QQ'], row['x_b'], row['t'], row['k'],row['phi_x'], row['varF']])
        pseudodata_df['set'].append(tempSet)
        pseudodata_df['k'].append(tempk)
        pseudodata_df['QQ'].append(tempQQ)
        pseudodata_df['x_b'].append(tempxb)
        pseudodata_df['t'].append(tempt)
        pseudodata_df['phi_x'].append(tempphi)
        ####################
        ReH=ReHps(tempxb,tempt)
        ReE=ReEps(tempxb,tempt)
        ReHtilde=ReHtps(tempxb,tempt)
        dvcs=DVCSps(tempxb,tempt)
        pseudodata_df['ReH'].append(ReH)
        pseudodata_df['ReE'].append(ReE)
        pseudodata_df['ReHt'].append(ReHtilde)
        pseudodata_df['dvcs'].append(dvcs)
        #print(pseudodata_df['dvcs'])
        ###################
        F1, F2 = fns.f1_f21(tempt)
        tempF = calc.fn_1([tempphi, tempQQ, tempxb, tempt, tempk, F1, F2], [ReH, ReE, ReHtilde, dvcs])
        #tempFerr = tempF * varF
        tempFerr = np.abs(tempF * varF) ## Had to do abs due to a run-time error
        #print(tempFerr)
        #pseudodata_df['F'].append(tempF)
        pseudodata_df['sigmaF'].append(tempFerr)
        SampleF = np.random.normal(loc=tempF, scale=tempFerr)
        #print(SampleF)
        pseudodata_df['F'].append(SampleF)
    return pd.DataFrame(pseudodata_df)



tempPseudoDatadf=GeneratePseudoData(kindf)
tempPseudoDatadf.to_csv('Pseudo_data_with_sampling.csv', index=False)

###### Making 3d scatter plots of CFFs ########

file_name = "Pseudo_data_with_sampling.csv"
df = pd.read_csv(file_name)

# Extract relevant columns
x_b_values = df['x_b'].values
t_values = df['t'].values

# Compute CFF values
ReH_values = df['ReH'].values
ReE_values = df['ReE'].values
ReHt_values = df['ReHt'].values
dvcs_values = df['dvcs'].values


##########################################
########## 3D surface plots ##############
##########################################

# Create a meshgrid for interpolation
grid_x, grid_t = np.meshgrid(
    np.linspace(x_b_values.min(), x_b_values.max(), 50),
    np.linspace(t_values.min(), t_values.max(), 50)
)

# Function to interpolate and create surface plots
def plot_surface(ax, x, y, z, title, zlabel, cmap='viridis'):
    grid_z = griddata((x, y), z, (grid_x, grid_t), method='cubic')  # Interpolation
    surf = ax.plot_surface(grid_x, grid_t, grid_z, cmap=cmap, edgecolor='none', alpha=0.8)
    ax.set_xlabel('x_b', fontsize=12)
    ax.set_ylabel('t', fontsize=12)
    ax.set_zlabel(zlabel, fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.view_init(elev=30, azim=45)  # Adjust view angle
    return surf

# Create figure
fig1 = plt.figure(figsize=(16, 12))
plt.suptitle("Basic Model 2: 3D Surface Plots of CFFs", fontsize=14, fontweight='bold')

# ReH surface plot
ax1 = fig1.add_subplot(221, projection='3d')
surf1 = plot_surface(ax1, x_b_values, t_values, ReH_values, '$ReH$ vs $x_B$ and $t$', '$ReH$')

# ReE surface plot
ax2 = fig1.add_subplot(222, projection='3d')
surf2 = plot_surface(ax2, x_b_values, t_values, ReE_values, '$ReE$ vs $x_B$ and $t$', '$ReE$')

# ReHt surface plot
ax3 = fig1.add_subplot(223, projection='3d')
surf3 = plot_surface(ax3, x_b_values, t_values, ReHt_values, '$Re\\tilde{H}$ vs $x_B$ and $t$', '$Re\\tilde{H}$')

# DVCS surface plot
ax4 = fig1.add_subplot(224, projection='3d')
surf4 = plot_surface(ax4, x_b_values, t_values, dvcs_values, '$DVCS$ vs $x_B$ and $t$', '$DVCS$')

# Add colorbars
fig1.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
fig1.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
fig1.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
fig1.colorbar(surf4, ax=ax4, shrink=0.5, aspect=10)

# Adjust layout and save figure
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("CFF_Surface_Plots.png", dpi=300)




##########################################
########## 3D scatter plots ##############
##########################################


vmin = min(ReH_values.min(), ReE_values.min(), ReHt_values.min(), dvcs_values.min())
vmax = max(ReH_values.max(), ReE_values.max(), ReHt_values.max(), dvcs_values.max())

# Create figure
fig2 = plt.figure(figsize=(16, 12))
plt.suptitle("Basic Model 2: 3D Scatter Plots of CFFs", fontsize=14, fontweight='bold')

# Function for creating subplots
def create_3d_subplot(ax, x, y, z, title, zlabel, cmap='viridis'):
    sc = ax.scatter(x, y, z, c=z, cmap=cmap, s=50, marker='o', vmin=vmin, vmax=vmax)
    ax.set_xlabel('x_b', fontsize=12)
    ax.set_ylabel('t', fontsize=12)
    ax.set_zlabel(zlabel, fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.view_init(elev=30, azim=45)  # Adjust view angle for better visibility
    return sc

# Create subplots
ax1 = fig2.add_subplot(221, projection='3d')
sc1 = create_3d_subplot(ax1, x_b_values, t_values, ReH_values, '$ReH$ vs $x_B$ and $t$', '$ReH$')

ax2 = fig2.add_subplot(222, projection='3d')
sc2 = create_3d_subplot(ax2, x_b_values, t_values, ReE_values, '$ReE$ vs $x_B$ and $t$', '$ReE$')

ax3 = fig2.add_subplot(223, projection='3d')
sc3 = create_3d_subplot(ax3, x_b_values, t_values, ReHt_values, '$Re\\tilde{H}$ vs $x_B$ and $t$', '$Re\\tilde{H}$')

ax4 = fig2.add_subplot(224, projection='3d')
sc4 = create_3d_subplot(ax4, x_b_values, t_values, dvcs_values, '$DVCS$ vs $x_B$ and $t$', '$DVCS$')

# Add colorbars
fig2.colorbar(sc1, ax=ax1, shrink=0.5, aspect=10)
fig2.colorbar(sc2, ax=ax2, shrink=0.5, aspect=10)
fig2.colorbar(sc3, ax=ax3, shrink=0.5, aspect=10)
fig2.colorbar(sc4, ax=ax4, shrink=0.5, aspect=10)

# Adjust layout and save figure
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("CFF_Scatter_Plots.png", dpi=300)