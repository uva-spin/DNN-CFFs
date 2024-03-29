import pandas as pd
import numpy as np
import tensorflow as tf
from BHDVCS_tf_modified import *

fns = F1F2()
calc = F_calc()

########## INPUT from the User ###################
interval = 10  # This the interval for angle (phi) binning
### Kinematics file ####
file_name = 'AllJlabData.csv'
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
    temp_f = -1.37
    return genf(x,t,temp_a, temp_b, temp_c, temp_d, temp_e, temp_f)

def ReEps(x,t):
    temp_a = 144.56
    temp_b = 149.99
    temp_c = 0.32
    temp_d = -1.09
    temp_e = -148.49
    temp_f = -0.31
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
    temp_a = 0.50
    temp_b = -0.41
    temp_c = 0.05
    temp_d = -0.25
    temp_e = 0.55
    temp_f = 0.166
    return genf(x,t,temp_a, temp_b, temp_c, temp_d, temp_e, temp_f)

## Note: sigmaF is considered as 5% of F ##

def GeneratePseudoData(df):
    pseudodata_df = {'k': [],
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
        #for phi in range(0, 361, interval):
        tempQQ, tempxb, tempt, tempk, tempPhi, tempSigmaF = np.array([row['QQ'], row['x_b'], row['t'], row['k'], row['phi_x'], row['sigmaF']])
        pseudodata_df['k'].append(tempk)
        pseudodata_df['QQ'].append(tempQQ)
        pseudodata_df['x_b'].append(tempxb)
        pseudodata_df['t'].append(tempt)
        pseudodata_df['phi_x'].append(tempPhi)
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
        tempF = calc.fn_1([tempPhi, tempQQ, tempxb, tempt, tempk, F1, F2], [ReH, ReE, ReHtilde, dvcs])
        pseudodata_df['F'].append(tempF)
        #pseudodata_df['sigmaF'].append(0.05*tempF)
        pseudodata_df['sigmaF'].append(tempSigmaF)
    return pd.DataFrame(pseudodata_df)

#print(GeneratePseudoData(kindf))
tempPseudoDatadf=GeneratePseudoData(kindf)
tempPseudoDatadf.to_csv('PseudoData_from_the_Basic_Model_for_JLab_Kinematics.csv', index=False)
