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
from definitions import *  # Replaces BHDVCS_tf_modified, km15, and dvcs_code - contains all classes
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import os

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")

sanity_checks_folder = 'Checks_for_Issues'
create_folders(sanity_checks_folder)

fns = F1F2()
calc = F_calc()

########## INPUT from the User ###################

### Kinematics file ####
file_name = 'AllJlabData_from_Zulkaida_and_Liliet.csv'
########## The input parameters for a, b, c, d, e, f need to be provided below #####

kindf = pd.read_csv(file_name, dtype=np.float32).dropna(axis=0, how='all').dropna(axis=1, how='all')

# Load km15_bkm10_dropped.csv for comparison
km15_bkm10_dropped_path = 'km15_bkm10_dropped.csv'
km15_bkm10_dropped = pd.read_csv(km15_bkm10_dropped_path, dtype=np.float32)

### Generating function (kept; no longer used for CFFs/DVCS) ###
# def genf(x,t,a,b,c,d,e,f):
#     temp = (a*(x**2) + b*x)*np.exp(c*(t**2)+d*t+e)+f
#     return temp

# # Kept for reference; no longer used to produce values
# def ReHps(x,t):
#     temp_a = -4.41
#     temp_b = 1.68
#     temp_c = -9.14
#     temp_d = -3.57
#     temp_e = 1.54
#     temp_f = 2.07
#     return genf(x,t,temp_a, temp_b, temp_c, temp_d, temp_e, temp_f)

# def ReEps(x,t):
#     temp_a = -1.04
#     temp_b = 0.46
#     temp_c = 0.6
#     temp_d = 1.95
#     temp_e = 2.72
#     temp_f = -0.95
#     return genf(x,t,temp_a, temp_b, temp_c, temp_d, temp_e, temp_f)

# def ReHtps(x,t):
#     temp_a = -1.86
#     temp_b = 1.50
#     temp_c = -0.29
#     temp_d = -1.33
#     temp_e = 0.46
#     temp_f = -0.98
#     return genf(x,t,temp_a, temp_b, temp_c, temp_d, temp_e, temp_f)

# def DVCSps(x,t):
#     temp_a = 0.52
#     temp_b = -0.41
#     temp_c = 0.05
#     temp_d = -0.25
#     temp_e = 0.55
#     temp_f = 0.173
#     return genf(x,t,temp_a, temp_b, temp_c, temp_d, temp_e, temp_f)

# This function returns a new DataFrame with kinematics if 'F' is negative.
def slice_negative_f(df):
    return df[df['F'] < 0]

# This function returns a new DataFrame with kinematics if 'F' > 20.
def slice_large_f(df):
    tempPseudoDatadf = df[df['F'] > 20.0]
    return pd.DataFrame(tempPseudoDatadf)

def GeneratePseudoData_noSampling(df):
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
        tempSet, tempQQ, tempxb, tempt, tempk, tempphi, varF = np.array(
            [row['#Set'],row['QQ'], row['x_b'], row['t'], row['k'],row['phi_x'], row['varF']]
        )
        pseudodata_df['set'].append(tempSet)
        pseudodata_df['k'].append(tempk)
        pseudodata_df['QQ'].append(tempQQ)
        pseudodata_df['x_b'].append(tempxb)
        pseudodata_df['t'].append(tempt)
        pseudodata_df['phi_x'].append(tempphi)

        # ============================= CHANGED: CFFs from KM15; DVCS from dvcs_code =============================
        # KM15 CFFs: ReH, ImH, ReE, ReHtilde, ImHtilde, ReEtilde
        ReH_KM15, ImH_KM15, ReE_KM15, ReHt_KM15, ImHt_KM15, ReEt_KM15 = ModKM15_CFFs(tempQQ, tempxb, tempt)

        # DVCS term from BKM10 via dvcs_code; set ImE=ImEt=0 as per your KM15 output
        phi_rad = np.deg2rad(tempphi)
        _, _, DVCS_term, _ = compute_quantities(
            tempk, tempQQ, tempxb, tempt, phi_rad,
            ReH_KM15, ReHt_KM15, ReE_KM15, ReEt_KM15, ImH_KM15, ImHt_KM15, 0.0, 0.0
        )
        dvcs_val = float(np.asarray(DVCS_term).squeeze())

        # record KM15 real CFFs and the BKM10 DVCS term
        pseudodata_df['ReH'].append(float(ReH_KM15))
        pseudodata_df['ReE'].append(float(ReE_KM15))
        pseudodata_df['ReHt'].append(float(ReHt_KM15))
        pseudodata_df['dvcs'].append(dvcs_val)
        # ========================================================================================================
        
        F1, F2 = fns.f1_f21(tempt)
        tempF = calc.fn_1([tempphi, tempQQ, tempxb, tempt, tempk, F1, F2],
                          [float(ReH_KM15), float(ReE_KM15), float(ReHt_KM15), dvcs_val])
        pseudodata_df['F'].append(tempF)
        tempFerr = np.abs(tempF * varF)
        
        # Debug print statements for sets 51 and 52
        if tempSet == 51 or tempSet == 52:
            csv_F = row['F']
            csv_sigmaF = row['sigmaF']
            csv_varF = varF
            expected_sigmaF = csv_F * csv_varF
            calculated_sigmaF = tempFerr
            difference = expected_sigmaF - csv_sigmaF
            
            # Find matching row in km15_bkm10_dropped.csv
            matching_rows = km15_bkm10_dropped[
                (km15_bkm10_dropped['set'] == tempSet) &
                (np.abs(km15_bkm10_dropped['QQ'] - tempQQ) < 1e-6) &
                (np.abs(km15_bkm10_dropped['x_b'] - tempxb) < 1e-6) &
                (np.abs(km15_bkm10_dropped['t'] - tempt) < 1e-6) &
                (np.abs(km15_bkm10_dropped['phi_x'] - tempphi) < 1e-6)
            ]
            
            if len(matching_rows) > 0:
                actual_sigmaF_km15 = matching_rows.iloc[0]['sigmaF']
                percent_diff = ((expected_sigmaF - actual_sigmaF_km15) / actual_sigmaF_km15 * 100) if actual_sigmaF_km15 != 0 else np.nan
            else:
                actual_sigmaF_km15 = None
                percent_diff = None
            
            print(f"\n=== Set {tempSet}, Row {i} ===")
            print(f"  F: {csv_F}")
            print(f"  sigmaF (from AllJlabData): {csv_sigmaF}")
            print(f"  calculated (tempF * varF): {calculated_sigmaF}")
            print(f"  subtraction: {row['F'] * row['varF'] - row['sigmaF']}")
            print(f"  difference (expected - actual): {difference}")
            if actual_sigmaF_km15 is not None:
                print(f"  sigmaF from km15_bkm10_dropped.csv: {actual_sigmaF_km15}")
                print(f"  percent difference (expected vs km15_bkm10_dropped): {percent_diff:.4f}%")
            else:
                print(f"  No matching row found in km15_bkm10_dropped.csv")
        
        pseudodata_df['sigmaF'].append(tempFerr)

    return pd.DataFrame(pseudodata_df)

tempPseudoData_FUNCTION_df = GeneratePseudoData_noSampling(kindf)
tempPseudoData_FUNCTION_df.to_csv(str(sanity_checks_folder)+'/'+'Pseudo_data_without_sampling.csv', index=False)

negative_F_kinematics_from_FUNCTION = slice_negative_f(tempPseudoData_FUNCTION_df)
negative_F_kinematics_from_FUNCTION.to_csv(str(sanity_checks_folder)+'/'+'ATTENTION_Kinematics_wth_negaive_F_from_FUNCTION.csv', index=False)

large_F_kinematics_from_FUNCTION = slice_large_f(tempPseudoData_FUNCTION_df)
large_F_kinematics_from_FUNCTION.to_csv(str(sanity_checks_folder)+'/'+'ATTENTION_Kinematics_wth_large_F_from_FUNCTION.csv', index=False)

print("************************* ATTENTION!!!  **************************************")
print(f"Please check the {sanity_checks_folder} folder to see which kinematic sets produce either very large cross-section values or negative cross-section values.")
print(" Note: In the files starts with filename *ATTENTION*, we haven't set F (cross-section) values to zero as it was done in the original data file AllJlabData_from_Zulkaida_and_Liliet.csv.")
print("In the 'Pseudo_data_with_sampling.csv' file, F values were set to zero if the correspondin ones were set to zero in the file AllJlabData_from_Zulkaida_and_Liliet.csv ")
print("******************************************************************************")

###### Making 3d scatter plots of CFFs ########
df = tempPseudoData_FUNCTION_df

# Extract relevant columns
x_b_values = df['x_b'].values
t_values = df['t'].values

# Compute CFF values
ReH_values = df['ReH'].values
ReE_values = df['ReE'].values
ReHt_values = df['ReHt'].values
dvcs_values = df['dvcs'].values

##########################################
########## 3D scatter plots ##############
##########################################

vmin = min(ReH_values.min(), ReE_values.min(), ReHt_values.min(), dvcs_values.min())
vmax = max(ReH_values.max(), ReE_values.max(), ReHt_values.max(), dvcs_values.max())

fig2 = plt.figure(figsize=(16, 12))
plt.suptitle("Basic Model 1: 3D Scatter Plots of CFFs", fontsize=14, fontweight='bold')

def create_3d_subplot(ax, x, y, z, title, zlabel, cmap='viridis'):
    sc = ax.scatter(x, y, z, c=z, cmap=cmap, s=50, marker='o', vmin=vmin, vmax=vmax)
    ax.set_xlabel('x_b', fontsize=12)
    ax.set_ylabel('t', fontsize=12)
    ax.set_zlabel(zlabel, fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.view_init(elev=30, azim=45)
    return sc

ax1 = fig2.add_subplot(221, projection='3d')
sc1 = create_3d_subplot(ax1, x_b_values, t_values, ReH_values, '$ReH$ vs $x_B$ and $t$', '$ReH$')

ax2 = fig2.add_subplot(222, projection='3d')
sc2 = create_3d_subplot(ax2, x_b_values, t_values, ReE_values, '$ReE$ vs $x_B$ and $t$', '$ReE$')

ax3 = fig2.add_subplot(223, projection='3d')
sc3 = create_3d_subplot(ax3, x_b_values, t_values, ReHt_values, '$Re\\tilde{H}$ vs $x_B$ and $t$', '$Re\\tilde{H}$')

ax4 = fig2.add_subplot(224, projection='3d')
sc4 = create_3d_subplot(ax4, x_b_values, t_values, dvcs_values, '$DVCS$ vs $x_B$ and $t$', '$DVCS$')

fig2.colorbar(sc1, ax=ax1, shrink=0.5, aspect=10)
fig2.colorbar(sc2, ax=ax2, shrink=0.5, aspect=10)
fig2.colorbar(sc3, ax=ax3, shrink=0.5, aspect=10)
fig2.colorbar(sc4, ax=ax4, shrink=0.5, aspect=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("CFF_Scatter_Plots.png", dpi=300)

##########################################
########## 3D surface plots ##############
##########################################

grid_x, grid_t = np.meshgrid(
    np.linspace(x_b_values.min(), x_b_values.max(), 50),
    np.linspace(t_values.min(), t_values.max(), 50)
)

def plot_surface(ax, x, y, z, title, zlabel, cmap='viridis'):
    grid_z = griddata((x, y), z, (grid_x, grid_t), method='cubic')
    surf = ax.plot_surface(grid_x, grid_t, grid_z, cmap=cmap, edgecolor='none', alpha=0.8)
    ax.set_xlabel('x_b', fontsize=12)
    ax.set_ylabel('t', fontsize=12)
    ax.set_zlabel(zlabel, fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.view_init(elev=30, azim=45)
    return surf

fig1 = plt.figure(figsize=(16, 12))
plt.suptitle("Basic Model 1: 3D Surface Plots of CFFs", fontsize=14, fontweight='bold')

ax1 = fig1.add_subplot(221, projection='3d')
surf1 = plot_surface(ax1, x_b_values, t_values, ReH_values, '$ReH$ vs $x_B$ and $t$', '$ReH$')

ax2 = fig1.add_subplot(222, projection='3d')
surf2 = plot_surface(ax2, x_b_values, t_values, ReE_values, '$ReE$ vs $x_B$ and $t$', '$ReE$')

ax3 = fig1.add_subplot(223, projection='3d')
surf3 = plot_surface(ax3, x_b_values, t_values, ReHt_values, '$Re\\tilde{H}$ vs $x_B$ and $t$', '$Re\\tilde{H}$')

ax4 = fig1.add_subplot(224, projection='3d')
surf4 = plot_surface(ax4, x_b_values, t_values, dvcs_values, '$DVCS$ vs $x_B$ and $t$', '$DVCS$')

fig1.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
fig1.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
fig1.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
fig1.colorbar(surf4, ax=ax4, shrink=0.5, aspect=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("CFF_Surface_Plots.png", dpi=300)

###########  Generating pseudo-data #####################

def GeneratePseudoData(df):
    pseudodata_df = {'set': [],
                     'k': [],
                     'QQ': [],
                     'x_b': [],
                     't': [],
                     'phi_x': [],
                     'F_function':[],
                     'F':[],
                     'sigmaF':[],                     
                     'ReH': [],
                     'ReE': [],
                     'ReHt': [],
                     'dvcs': []}
    
    # Handle set 55 separately: generate 1000 points with parabola-like F vs phi shape
    set_55_rows = df[df['#Set'] == 55]
    if len(set_55_rows) > 0:
        # Get kinematics from first row of set 55
        first_row_55 = set_55_rows.iloc[0]
        tempQQ_55 = first_row_55['QQ']
        tempxb_55 = first_row_55['x_b']
        tempt_55 = first_row_55['t']
        tempk_55 = first_row_55['k']
        avg_varF_55 = set_55_rows['varF'].mean()
        
        # Calculate CFFs (same for all phi values)
        ReH_KM15, ImH_KM15, ReE_KM15, ReHt_KM15, ImHt_KM15, ReEt_KM15 = ModKM15_CFFs(tempQQ_55, tempxb_55, tempt_55)
        F1, F2 = fns.f1_f21(tempt_55)
        
        # Calculate F values for original phi points to fit parabola
        phi_original = set_55_rows['phi_x'].values
        F_original = []
        for tempphi in phi_original:
            phi_rad = np.deg2rad(tempphi)
            _, _, DVCS_term, _ = compute_quantities(
                tempk_55, tempQQ_55, tempxb_55, tempt_55, phi_rad,
                ReH_KM15, ReHt_KM15, ReE_KM15, ReEt_KM15, ImH_KM15, ImHt_KM15, 0.0, 0.0
            )
            dvcs_val = float(np.asarray(DVCS_term).squeeze())
            tempF = calc.fn_1([tempphi, tempQQ_55, tempxb_55, tempt_55, tempk_55, F1, F2],
                              [float(ReH_KM15), float(ReE_KM15), float(ReHt_KM15), dvcs_val])
            F_original.append(tempF)
        
        F_original = np.array(F_original)
        
        # Fit a parabola: F = a*phi^2 + b*phi + c
        # Convert phi to radians for better numerical stability, or use degrees
        phi_deg = phi_original
        # Fit polynomial of degree 2
        poly_coeffs = np.polyfit(phi_deg, F_original, 2)
        parabola_func = np.poly1d(poly_coeffs)
        
        # Generate 1000 evenly spaced phi_x values from 0 to 360 degrees
        phi_x_1000 = np.linspace(0, 360, 1000, endpoint=False)
        
        # Generate data for set 55 with 1000 points using parabola fit
        for tempphi in phi_x_1000:
            pseudodata_df['set'].append(55)
            pseudodata_df['k'].append(tempk_55)
            pseudodata_df['QQ'].append(tempQQ_55)
            pseudodata_df['x_b'].append(tempxb_55)
            pseudodata_df['t'].append(tempt_55)
            pseudodata_df['phi_x'].append(tempphi)
            
            # Calculate CFFs and DVCS (same CFFs, but DVCS depends on phi)
            phi_rad = np.deg2rad(tempphi)
            _, _, DVCS_term, _ = compute_quantities(
                tempk_55, tempQQ_55, tempxb_55, tempt_55, phi_rad,
                ReH_KM15, ReHt_KM15, ReE_KM15, ReEt_KM15, ImH_KM15, ImHt_KM15, 0.0, 0.0
            )
            dvcs_val = float(np.asarray(DVCS_term).squeeze())
            
            pseudodata_df['ReH'].append(float(ReH_KM15))
            pseudodata_df['ReE'].append(float(ReE_KM15))
            pseudodata_df['ReHt'].append(float(ReHt_KM15))
            pseudodata_df['dvcs'].append(dvcs_val)
            
            # Use parabola to get F value
            tempF = parabola_func(tempphi)
            # Ensure F is non-negative
            if tempF < 0:
                tempF = 0.0
            # For set 55: zero error, use exact F value (no sampling)
            tempFerr = 0.0
            SampleF = tempF  # Use exact value, no randomness
            
            pseudodata_df['F_function'].append(tempF)
            pseudodata_df['sigmaF'].append(tempFerr)
            pseudodata_df['F'].append(SampleF)
        
        # Remove set 55 from original df to avoid duplicates
        df = df[df['#Set'] != 55].reset_index(drop=True)
    
    # Process all other sets normally
    for i in range(len(df)):
        row = df.loc[i]
        tempSet, tempQQ, tempxb, tempt, tempk, tempphi, varF = np.array(
            [row['#Set'],row['QQ'], row['x_b'], row['t'], row['k'],row['phi_x'], row['varF']]
        )
        pseudodata_df['set'].append(tempSet)
        pseudodata_df['k'].append(tempk)
        pseudodata_df['QQ'].append(tempQQ)
        pseudodata_df['x_b'].append(tempxb)
        pseudodata_df['t'].append(tempt)
        pseudodata_df['phi_x'].append(tempphi)

        # ============================= CHANGED: CFFs from KM15; DVCS from dvcs_code =============================
        ReH_KM15, ImH_KM15, ReE_KM15, ReHt_KM15, ImHt_KM15, ReEt_KM15 = ModKM15_CFFs(tempQQ, tempxb, tempt)
        phi_rad = np.deg2rad(tempphi)
        _, _, DVCS_term, _ = compute_quantities(
            tempk, tempQQ, tempxb, tempt, phi_rad,
            ReH_KM15, ReHt_KM15, ReE_KM15, ReEt_KM15, ImH_KM15, ImHt_KM15, 0.0, 0.0
        )
        dvcs_val = float(np.asarray(DVCS_term).squeeze())

        pseudodata_df['ReH'].append(float(ReH_KM15))
        pseudodata_df['ReE'].append(float(ReE_KM15))
        pseudodata_df['ReHt'].append(float(ReHt_KM15))
        pseudodata_df['dvcs'].append(dvcs_val)
        # ========================================================================================================

        F1, F2 = fns.f1_f21(tempt)
        if 'sigmaF' in row and row['sigmaF'] == 0.0:
            tempF = 0.0
            tempFerr = np.abs(tempF * varF)
            SampleF = 0.0
        else:
            tempF = calc.fn_1([tempphi, tempQQ, tempxb, tempt, tempk, F1, F2],
                              [float(ReH_KM15), float(ReE_KM15), float(ReHt_KM15), dvcs_val])
            tempFerr = np.abs(tempF * varF)
            while True:
                SampleF = np.random.normal(loc=tempF, scale=tempFerr)
                if np.all(SampleF > 0):
                    break

        # Debug print statements for sets 51 and 52
        if tempSet == 51 or tempSet == 52:
            csv_F = row['F']
            csv_sigmaF = row['sigmaF'] if 'sigmaF' in row else None
            csv_varF = varF
            expected_sigmaF = csv_F * csv_varF
            calculated_sigmaF = tempFerr
            difference = expected_sigmaF - csv_sigmaF if csv_sigmaF is not None else None
            
            # Find matching row in km15_bkm10_dropped.csv
            matching_rows = km15_bkm10_dropped[
                (km15_bkm10_dropped['set'] == tempSet) &
                (np.abs(km15_bkm10_dropped['QQ'] - tempQQ) < 1e-6) &
                (np.abs(km15_bkm10_dropped['x_b'] - tempxb) < 1e-6) &
                (np.abs(km15_bkm10_dropped['t'] - tempt) < 1e-6) &
                (np.abs(km15_bkm10_dropped['phi_x'] - tempphi) < 1e-6)
            ]
            
            if len(matching_rows) > 0:
                actual_sigmaF_km15 = matching_rows.iloc[0]['sigmaF']
                percent_diff = ((expected_sigmaF - actual_sigmaF_km15) / actual_sigmaF_km15 * 100) if actual_sigmaF_km15 != 0 else np.nan
            else:
                actual_sigmaF_km15 = None
                percent_diff = None
            
            print(f"\n=== GeneratePseudoData: Set {tempSet}, Row {i} ===")
            print(f"  F: {csv_F}")
            print(f"  sigmaF (from AllJlabData): {csv_sigmaF}")
            print(f"  calculated (tempF * varF): {calculated_sigmaF}")
            if csv_sigmaF is not None:
                print(f"  subtraction: {row['F'] * row['varF'] - row['sigmaF']}")
            if difference is not None:
                print(f"  difference (expected - actual): {difference}")
            if actual_sigmaF_km15 is not None:
                print(f"  sigmaF from km15_bkm10_dropped.csv: {actual_sigmaF_km15}")
                print(f"  percent difference (expected vs km15_bkm10_dropped): {percent_diff:.4f}%")
            else:
                print(f"  No matching row found in km15_bkm10_dropped.csv")

        pseudodata_df['F_function'].append(tempF)
        pseudodata_df['sigmaF'].append(tempFerr)
        pseudodata_df['F'].append(SampleF)

    return pd.DataFrame(pseudodata_df)

tempPseudoDatadf = GeneratePseudoData(kindf)
tempPseudoDatadf.to_csv('Pseudo_data_with_sampling.csv', index=False)
# Also output as km15_bkm10_dropped.csv for use in subsequent steps
tempPseudoDatadf.to_csv('km15_bkm10_dropped.csv', index=False)

negative_F_kinematics = slice_negative_f(tempPseudoDatadf)
negative_F_kinematics.to_csv(str(sanity_checks_folder)+'/'+'Kinematics_wth_negaive_F_after_sampling.csv', index=False)

large_F_kinematics = slice_large_f(tempPseudoDatadf)
large_F_kinematics.to_csv(str(sanity_checks_folder)+'/'+'Kinematics_wth_large_F_after_sampling.csv', index=False)
