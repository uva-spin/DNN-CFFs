import numpy as np
import pandas as pd
import tensorflow as tf
from BHDVCS_tf_modified import *
import matplotlib.pyplot as plt
import os
import sys


# if len(sys.argv) != 2:
#     raise ValueError(f"Usage: python script.py arg1") 
# kinematicSet = int(sys.argv[1])
kinematicSet = 0
data_file = 'Work\Aaryan\PseudoData_from_the_Basic_Model_for_JLab_Kinematics.csv'
df = pd.read_csv(data_file, dtype=np.float64)
kinematicDf = df.iloc[kinematicSet*24: (kinematicSet+1)*24]

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")
        
def find_intersection(*equations):
    # Create a matrix A and a vector B from the equations
    A = np.array([eq[:-1] for eq in equations])
    B = -np.array([eq[-1] for eq in equations])

    # Solve the system of linear equations
    intersection_point = np.linalg.solve(A, B)

    return tuple(intersection_point)

def find_f(reh, ree, df, angle, reht = []):
    row = df.loc[df['phi_x'] == angle]
    input_tensor= tf.cast([[row['QQ'].iloc[0], row['x_b'].iloc[0], row['t'].iloc[0], angle, row['k'].iloc[0]]], dtype=tf.float32) 	

    correct_f = row['F'].iloc[0]
    dvcs = row['dvcs'].iloc[0]
    f_values = []
    bhdvcstf = BHDVCStf() 

    if len(reht) > 0: #3D
        for i in range(len(reh)):
            params = tf.cast([[reh[i], ree[i], reht[i], dvcs]], dtype=tf.float32) # h, e, ht, dvcs

            pred_f = bhdvcstf.curve_fit(input_tensor, params)
            f_values.append(abs(correct_f - pred_f))
    else: #2D
        reht = row['ReHt'].iloc[0]
        for i in range(len(reh)):
            params = tf.cast([[reh[i], ree[i], reht, dvcs]], dtype=tf.float32) # h, e, ht, dvcs

            pred_f = bhdvcstf.curve_fit(input_tensor, params)
            f_values.append(abs(correct_f - pred_f))


    return f_values

def get_min_coords(f, *coords):
    min_indices = np.argsort(np.array(f).flatten())[:10]
    coordinates = []
    for i in min_indices: 
        coord = []
        for c in coords[0]: 
            coord.append(c[i])
        coordinates.append(coord)

    return coordinates

def exponential_normalize_and_scale(values, scale_factor=1.0):
    values = np.array(values)

    # Exponential normalization
    normalized_values = scale_factor * (np.exp(values) - 1)

    # Scaling to the range of 1 to 100
    min_val = min(normalized_values)
    max_val = max(normalized_values)
    scaled_values = 1 + (normalized_values - min_val) * (99 / (max_val - min_val))

    return scaled_values

def combined_graph(f, cffs, angles):
    plt.clf()
    if len(cffs) == 2: 
        f1, f2 = f
        reh, ree = cffs
        angle1, angle2 = angles

        min_coords = get_min_coords(f1, [reh, ree])
        x, y = zip(*min_coords)
        m, b = np.polyfit(x, y, 1)

        min_coords = get_min_coords(f2, [reh, ree])
        x, y = zip(*min_coords)
        m1, b1 = np.polyfit(x, y, 1)

        line_y = m * reh + b
        plt.plot(reh, line_y, color='yellow')
        line_y = m1 * reh + b1
        plt.plot(reh, line_y, color='yellow')

        intX, intY = find_intersection([-m, 1, -b], [-m1, 1, -b1]) # y = mx + b -> -mx + y - b = 0

        normf1 = exponential_normalize_and_scale(f1)
        normf2 = exponential_normalize_and_scale(f2)
        sc = plt.scatter(reh, ree, c=-1*np.add(normf1,normf2), cmap = 'RdYlGn')

        text_coordinates = f'({round(intX, 2)}, {round(intY, 2)})'
        plt.text(intX + 0.8, intY + 0.8, text_coordinates, fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
        plt.xlim(-10,10)
        plt.ylim(-10,10)
        plt.title(f'Loss Plot ReH/ReE on Set {kinematicSet}')
        plt.xlabel('ReH')
        plt.ylabel('ReE')
        plt.savefig(f'Work/Aaryan/2D Surface Plots/Combined_Set{kinematicSet}_{angle1}_{angle2}.png')
    if len(cffs) == 3: 
        #TODO 
        pass

def save_graph(f, angle, cffs):
    plt.clf()
    if len(cffs) == 2: 
        reh, ree = cffs
        sc = plt.scatter(reh, ree, c=-1*np.array(f), cmap = 'RdYlGn')
        plt.colorbar(sc)
        plt.title(f'2D Loss Plot at {angle}')
        plt.xlabel('ReH')
        plt.ylabel('ReE')
        plt.savefig(f'Work/Aaryan/2D Surface Plots/lossPlot_Set{kinematicSet}_{angle}.png')
    else: 
        reh, ree, reht = cffs
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(reh, ree, reht, c=-1*np.array(f), cmap='RdYlGn', alpha=0.3)
        plt.colorbar(scatter)
        ax.set_title(f'3D Loss Plot at {angle}')
        ax.set_xlabel('ReH')
        ax.set_ylabel('ReE')
        ax.set_zlabel('ReHt')
        plt.savefig(f'Work/Aaryan/3D Surface Plots/lossPlot_Set{kinematicSet}_{angle}.png')

def create_2D_lossPlot(df, angle1 = 7.5, angle2 = 187.5):
    cff_range = np.linspace(-10, 10, 50)
    grid = np.meshgrid(cff_range, cff_range)
    
    reh, ree = grid
    reh = reh.flatten()
    ree = ree.flatten()

    f1 = find_f(reh, ree, df, angle1)
    f2 = find_f(reh, ree, df, angle2)

    save_graph(f1, angle1, [reh, ree])
    save_graph(f2, angle2, [reh, ree])

    combined_graph([f1,f2], [reh,ree], [angle1,angle2])

def find_eqn_3D(f, reh, ree, reht):
    x, y, z = zip(*get_min_coords(f, [reh, ree, reht]))
    design_matrix = np.column_stack((x, y, np.ones_like(x)))
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, z, rcond=None)
    a, b, c = coefficients
    return [a, b, -1, c]

def extractCFF_3D(df, angle1 = 7.5, angle2 = 127.5, angle3 = 247.5):
    cff_range = np.linspace(-1.2, 0.2, 35)
    grid = np.meshgrid(cff_range, cff_range, cff_range)

    reh, ree, reht = grid
    reh = reh.flatten()
    ree = ree.flatten()
    reht = reht.flatten()

    f1 = find_f(reh, ree, df, angle1, reht)
    f2 = find_f(reh, ree, df, angle2, reht)
    f3 = find_f(reh, ree, df, angle3, reht)

    save_graph(f1, angle1, [reh, ree, reht])
    save_graph(f2, angle2, [reh, ree, reht])
    save_graph(f3, angle3, [reh, ree, reht])

    eqn1 = find_eqn_3D(f1, reh, ree, reht)
    eqn2 = find_eqn_3D(f2, reh, ree, reht)
    eqn3 = find_eqn_3D(f3, reh, ree, reht)

    print(find_intersection(eqn1, eqn2, eqn3))

create_folders('2D Surface Plots')
create_folders('3D Surface Plots')

# create_2D_lossPlot(kinematicDf, angle1 = 7.5, angle2 = 187.5)
extractCFF_3D(kinematicDf, angle1 = 7.5, angle2 = 127.5, angle3 = 247.5)

 

