##########################################################################
###############  Written by Ishara Fernando                  #############
##############  Revised and Extended by Ani                  #############
##########################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BHDVCS_without_tf_for_sampling import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import os
import cppyy
import math
from scipy.integrate import quad
from math import pi, sqrt

# ROOT class import
cppyy.include("TBKM.h")
cppyy.load_library("TBKM_cxx.so")  # Ensure this points to the correct path
TBKM = cppyy.gbl.TBKM
TComplex = cppyy.gbl.TComplex

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

file_name = 'AllJlabData_from_Zulkaida_and_Liliet.csv'
kindf = pd.read_csv(file_name, dtype=np.float32).dropna(axis=0, how='all').dropna(axis=1, how='all')

# KM15 parameterized CFFs
def ModKM15_CFFs(QQ, xB, t, k=0.0):
    nval, pval, nsea, rsea, psea, bsea = 1.35, 1., 1.5, 1., 2., 4.6
    Mval, rval, bval, C0, Msub = 0.789, 0.918, 0.4, 2.768, 1.204
    Mtval, rtval, btval, ntval = 3.993, 0.881, 0.4, 0.6
    Msea, rpi, Mpi = sqrt(0.482), 2.646, 4.

    alpha_val = 0.43 + 0.85 * t
    alpha_sea = 1.13 + 0.15 * t
    Ct = C0 / pow(1. - t / Msub / Msub, 2.)
    xi = xB / (2. - xB)

    def fHval(x):
        return (nval * rval) / (1. + x) * pow((2. * x) / (1. + x), -alpha_val) * \
               pow((1. - x) / (1. + x), bval) / \
               pow(1. - ((1. - x) / (1. + x)) * (t / Mval / Mval), pval)

    def fHsea(x):
        return (nsea * rsea) / (1. + x) * pow((2. * x) / (1. + x), -alpha_sea) * \
               pow((1. - x) / (1. + x), bsea) / \
               pow(1. - ((1. - x) / (1. + x)) * (t / Msea / Msea), psea)

    def fHtval(x):
        return (ntval * rtval) / (1. + x) * pow((2. * x) / (1. + x), -alpha_val) * \
               pow((1. - x) / (1. + x), btval) / \
               (1. - ((1. - x) / (1. + x)) * (t / Mtval / Mtval))

    def fImH(x): return pi * ((2. * (4. / 9.) + 1. / 9.) * fHval(x) + 2. / 9. * fHsea(x))
    def fImHt(x): return pi * (2. * (4. / 9.) + 1. / 9.) * fHtval(x)
    def fPV_ReH(x): return -2. * x / (x + xi) * fImH(x)
    def fPV_ReHt(x): return -2. * xi / (x + xi) * fImHt(x)

    DR_ReH, _ = quad(fPV_ReH, 1e-6, 1.0, weight='cauchy', wvar=xi)
    DR_ReHt, _ = quad(fPV_ReHt, 1e-6, 1.0, weight='cauchy', wvar=xi)

    ImH = fImH(xi)
    ReH = 1. / pi * DR_ReH - Ct
    ReE = Ct
    ImHt = fImHt(xi)
    ReHt = 1. / pi * DR_ReHt
    ReEt = rpi / xi * 2.164 / ((0.0196 - t) * pow(1. - t / Mpi / Mpi, 2.))

    return ReH, ImH, ReE, ReHt, ImHt, ReEt

def slice_negative_f(df):
    return df[df['F'] < 0]

def slice_large_f(df):
    return df[df['F'] > 20.0]

def GeneratePseudoData_noSampling(df):
    tbkm = TBKM()
    pseudodata_df = {'set': [], 'k': [], 'QQ': [], 'x_b': [], 't': [], 'phi_x': [],
                     'F': [], 'sigmaF': [], 'ReH': [], 'ReE': [], 'ReHt': [], 'dvcs': []}

    for i in range(len(df)):
        row = df.loc[i]
        tempSet, tempQQ, tempxb, tempt, tempk, tempphi, varF = \
            np.array([row['#Set'], row['QQ'], row['x_b'], row['t'], row['k'], row['phi_x'], row['varF']])
        pseudodata_df['set'].append(tempSet)
        pseudodata_df['k'].append(tempk)
        pseudodata_df['QQ'].append(tempQQ)
        pseudodata_df['x_b'].append(tempxb)
        pseudodata_df['t'].append(tempt)
        pseudodata_df['phi_x'].append(tempphi)

        ReH, ImH, ReE, ReHt, ImHt, ReEt = ModKM15_CFFs(tempQQ, tempxb, tempt)
        pseudodata_df['ReH'].append(ReH)
        pseudodata_df['ReE'].append(ReE)
        pseudodata_df['ReHt'].append(ReHt)

        t2cffs = cppyy.gbl.std.vector[TComplex]()
        t2cffs.push_back(TComplex(ReH, ImH))
        t2cffs.push_back(TComplex(ReE, 0.0))
        t2cffs.push_back(TComplex(ReHt, ImHt))
        t2cffs.push_back(TComplex(ReEt, 0.0))

        # This is the important part: pass pointer to the start of the vector
        tbkm.SetCFFs(t2cffs.data())


        kine_vec = cppyy.gbl.std.vector['double']()
        kine_vec += [tempQQ, tempxb, tempt, tempk]
        tbkm.SetKinematics(kine_vec.data())


        F1, F2 = fns.f1_f21(tempt)
        dvcs = tbkm.DVCS_UU_10(kine_vec.data(), tempphi, t2cffs.data(), "t2")
        pseudodata_df['dvcs'].append(dvcs)

        tempF = calc.fn_1([tempphi, tempQQ, tempxb, tempt, tempk, F1, F2], [ReH, ReE, ReHt, dvcs])
        pseudodata_df['F'].append(tempF)
        tempFerr = np.abs(tempF * varF)
        pseudodata_df['sigmaF'].append(tempFerr)

    return pd.DataFrame(pseudodata_df)

tempPseudoData_FUNCTION_df = GeneratePseudoData_noSampling(kindf)
tempPseudoData_FUNCTION_df.to_csv(str(sanity_checks_folder) + '/' + 'Pseudo_data_without_sampling_km15.csv', index=False)

negative_F_kinematics_from_FUNCTION = slice_negative_f(tempPseudoData_FUNCTION_df)
negative_F_kinematics_from_FUNCTION.to_csv(str(sanity_checks_folder) + '/' + 'ATTENTION_Kinematics_wth_negaive_F_from_FUNCTION.csv', index=False)

large_F_kinematics_from_FUNCTION = slice_large_f(tempPseudoData_FUNCTION_df)
large_F_kinematics_from_FUNCTION.to_csv(str(sanity_checks_folder) + '/' + 'ATTENTION_Kinematics_wth_large_F_from_FUNCTION.csv', index=False)

print("************************* ATTENTION!!!  **************************************")
print(f"Please check the {sanity_checks_folder} folder to see which kinematic sets produce either very large cross-section values or negative cross-section values.")
print("******************************************************************************")

##########################
## Additional Sampling-Based Generation + Plots
##########################

print("Generating pseudo-data with sampling...")

def GeneratePseudoData(df):
    # Create TBKM instance
    tbkm = cppyy.gbl.TBKM()

    pseudodata_df = {
        'set': [], 'k': [], 'QQ': [], 'x_b': [], 't': [], 'phi_x': [],
        'F_function': [], 'F': [], 'sigmaF': [],
        'ReH': [], 'ReE': [], 'ReHt': [], 'dvcs': []
    }

    for i in range(len(df)):
        row = df.loc[i]

        # Extract relevant variables from the row
        tempSet, tempQQ, tempxb, tempt, tempk, tempphi, varF = np.array([
            row['#Set'], row['QQ'], row['x_b'], row['t'], row['k'], row['phi_x'], row['varF']
        ])
        pseudodata_df['set'].append(tempSet)
        pseudodata_df['k'].append(tempk)
        pseudodata_df['QQ'].append(tempQQ)
        pseudodata_df['x_b'].append(tempxb)
        pseudodata_df['t'].append(tempt)
        pseudodata_df['phi_x'].append(tempphi)

        # Get the CFFs from KM15 model
        ReH, ImH, ReE, ReHt, ImHt, ReEt = ModKM15_CFFs(tempQQ, tempxb, tempt)

        # Create vector of TComplex CFFs
        t2cffs = cppyy.gbl.std.vector[cppyy.gbl.TComplex]()
        t2cffs.push_back(cppyy.gbl.TComplex(ReH, ImH))
        t2cffs.push_back(cppyy.gbl.TComplex(ReE, 0.0))
        t2cffs.push_back(cppyy.gbl.TComplex(ReHt, ImHt))
        t2cffs.push_back(cppyy.gbl.TComplex(ReEt, 0.0))

        # Set the CFFs
        tbkm.SetCFFs(t2cffs.data())

        # Kinematics vector
        kine_vec = cppyy.gbl.std.vector['double']()
        kine_vec += [tempQQ, tempxb, tempt, tempk]
        tbkm.SetKinematics(kine_vec.data())



        # Compute DVCS cross section
        dvcs = tbkm.DVCS_UU_10(kine_vec.data(), tempphi, t2cffs.data(), "t2")

        # Save real parts of CFFs and dvcs
        pseudodata_df['ReH'].append(ReH)
        pseudodata_df['ReE'].append(ReE)
        pseudodata_df['ReHt'].append(ReHt)
        pseudodata_df['dvcs'].append(dvcs)

        # Compute F and its uncertainty
        F1, F2 = fns.f1_f21(tempt)
        if row['sigmaF'] == 0.0:
            tempF = 0.0
            tempFerr = 0.0
            SampleF = 0.0
        else:
            tempF = calc.fn_1([tempphi, tempQQ, tempxb, tempt, tempk, F1, F2],
                              [ReH, ReE, ReHt, dvcs])
            tempFerr = np.abs(tempF * varF)
            attempts = 0
            while True:
                SampleF = np.random.normal(loc=tempF, scale=tempFerr)
                attempts += 1
                if SampleF > 0:
                    break
                if attempts > 10:
                    print(f"High sampling count at row {i}: {attempts} attempts, tempF={tempF}, tempFerr={tempFerr}")


        pseudodata_df['F_function'].append(tempF)
        pseudodata_df['sigmaF'].append(tempFerr)
        pseudodata_df['F'].append(SampleF)

    return pd.DataFrame(pseudodata_df)


tempPseudoDatadf = GeneratePseudoData(kindf)
tempPseudoDatadf.to_csv('Pseudo_data_with_sampling_km15.csv', index=False)

negative_F_kinematics = slice_negative_f(tempPseudoDatadf)
negative_F_kinematics.to_csv(os.path.join(sanity_checks_folder, 'Kinematics_wth_negaive_F_after_sampling.csv'), index=False)

large_F_kinematics = slice_large_f(tempPseudoDatadf)
large_F_kinematics.to_csv(os.path.join(sanity_checks_folder, 'Kinematics_wth_large_F_after_sampling.csv'), index=False)

##########################
## 3D Plotting
##########################

print("Creating 3D plots...")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

df = tempPseudoDatadf
x_b_values = df['x_b'].values
t_values = df['t'].values

ReH_values = df['ReH'].values
ReE_values = df['ReE'].values
ReHt_values = df['ReHt'].values
dvcs_values = df['dvcs'].values

vmin = min(ReH_values.min(), ReE_values.min(), ReHt_values.min(), dvcs_values.min())
vmax = max(ReH_values.max(), ReE_values.max(), ReHt_values.max(), dvcs_values.max())

# --- 3D Scatter Plots ---
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

# --- 3D Surface Plots ---
fig1 = plt.figure(figsize=(16, 12))
plt.suptitle("Basic Model 1: 3D Surface Plots of CFFs", fontsize=14, fontweight='bold')

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

print("Plotting complete.")
