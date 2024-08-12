import pandas as pd
import numpy as np
from BHDVCS import *

# Initialize the necessary objects
fns = F1F2()
calc = F_calc()

# Load the kinematics data
file_name = 'AllJlabData_from_Zulkaida_and_Liliet.csv'
kindf = pd.read_csv(file_name, dtype=np.float32).dropna(axis=0, how='all').dropna(axis=1, how='all')

# Define the generating function
def genf(x, t, a, b, c, d, e, f):
    return (a * x**2 + b * x) * np.exp(c * t**2 + d * t + e) + f

# Predefine the parameters for each function to avoid repeated calculations
params = {
    'ReHps': (-4.41, 1.68, -9.14, -3.57, 1.54, -1.37),
    'ReEps': (144.56, 149.99, 0.32, -1.09, -148.49, -0.31),
    'ReHtps': (-1.86, 1.50, -0.29, -1.33, 0.46, -0.98),
    'DVCSps': (0.50, -0.41, 0.05, -0.25, 0.55, 0.166)
}

# Vectorized functions
def ReHps(x, t): return genf(x, t, *params['ReHps'])
def ReEps(x, t): return genf(x, t, *params['ReEps'])
def ReHtps(x, t): return genf(x, t, *params['ReHtps'])
def DVCSps(x, t): return genf(x, t, *params['DVCSps'])

# Generate pseudo-data
def GeneratePseudoData(df):
    pseudodata_df = {
        'Set #': df['#Set'].values,
        'k': df['k'].values,
        'QQ': df['QQ'].values,
        'x_b': df['x_b'].values,
        't': df['t'].values,
        'phi_x': df['phi_x'].values,
        'F': [],
        'sigmaF': [],
        'ReH': ReHps(df['x_b'], df['t']),
        'ReE': ReEps(df['x_b'], df['t']),
        'ReHt': ReHtps(df['x_b'], df['t']),
        'dvcs': DVCSps(df['x_b'], df['t'])
    }
    
    for i in range(len(df)):
        kins = [
            df['phi_x'].iloc[i],
            df['QQ'].iloc[i],
            df['x_b'].iloc[i],
            df['t'].iloc[i],
            df['k'].iloc[i],
            *fns.f1_f21(df['t'].iloc[i])
        ]
        
        cffs = [
            pseudodata_df['ReH'][i],
            pseudodata_df['ReE'][i],
            pseudodata_df['ReHt'][i],
            pseudodata_df['dvcs'][i]
        ]
        
        tempF = calc.fn_1(kins, cffs)
        pseudodata_df['F'].append(tempF)
        pseudodata_df['sigmaF'].append(tempF * df['varF'].iloc[i])
    
    return pd.DataFrame(pseudodata_df)

# Generate and save the pseudo-data
tempPseudoDatadf = GeneratePseudoData(kindf)
tempPseudoDatadf.to_csv('Basic_out.csv', index=False)
