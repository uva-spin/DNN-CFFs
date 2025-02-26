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
        #SampleF = np.random.normal(loc=tempF, scale=tempFerr)
        while True:
            SampleF = np.random.normal(loc=tempF, scale=tempFerr)
            if SampleF > 0:
                break
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

# Create a single figure for all 4 scatter plots
fig = plt.figure(figsize=(16, 12))

# ReH scatter plot
ax1 = fig.add_subplot(221, projection='3d')
sc1 = ax1.scatter(x_b_values, t_values, ReH_values, c=ReH_values, marker='o')
ax1.set_xlabel('x_b')
ax1.set_ylabel('t')
ax1.set_zlabel('ReH')
ax1.set_title('ReH vs x_b and t')
fig.colorbar(sc1, ax=ax1, shrink=0.5)

# ReE scatter plot
ax2 = fig.add_subplot(222, projection='3d')
sc2 = ax2.scatter(x_b_values, t_values, ReE_values, c=ReE_values, marker='o')
ax2.set_xlabel('x_b')
ax2.set_ylabel('t')
ax2.set_zlabel('ReE')
ax2.set_title('ReE vs x_b and t')
fig.colorbar(sc2, ax=ax2, shrink=0.5)

# ReHt scatter plot
ax3 = fig.add_subplot(223, projection='3d')
sc3 = ax3.scatter(x_b_values, t_values, ReHt_values, c=ReHt_values, marker='o')
ax3.set_xlabel('x_b')
ax3.set_ylabel('t')
ax3.set_zlabel('ReHt')
ax3.set_title('ReHt vs x_b and t')
fig.colorbar(sc3, ax=ax3, shrink=0.5)

# DVCS scatter plot
ax4 = fig.add_subplot(224, projection='3d')
sc4 = ax4.scatter(x_b_values, t_values, dvcs_values, c=dvcs_values, marker='o')
ax4.set_xlabel('x_b')
ax4.set_ylabel('t')
ax4.set_zlabel('DVCS')
ax4.set_title('DVCS vs x_b and t')
fig.colorbar(sc4, ax=ax4, shrink=0.5)

# Save all scatter plots into a single PNG file
plt.tight_layout()
plt.savefig("CFF_Scatter_Plots.png", dpi=300)

# Close the plots to free memory
plt.close(fig)

print("Combined 3D scatter plots saved as 'CFF_Scatter_Plots.png'.")
