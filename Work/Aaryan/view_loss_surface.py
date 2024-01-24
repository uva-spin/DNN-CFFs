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

def find_f_2D(reh, ree, df, angle):
    row = df.loc[df['phi_x'] == angle]
    input_tensor= tf.cast([[row['QQ'].iloc[0], row['x_b'].iloc[0], row['t'].iloc[0], angle, row['k'].iloc[0]]], dtype=tf.float32) 	

    correct_f = row['F'].iloc[0]
    f_values = []
    bhdvcstf = BHDVCStf() 
    for i in range(len(reh)):
        params = tf.cast([[reh[i], ree[i], row['ReHt'].iloc[0], row['dvcs'].iloc[0]]], dtype=tf.float32) # h, e, ht, dvcs

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

def create_2D_lossPlot(df, angle1, angle2):
    cff_range = np.linspace(-10, 10, 50)
    grid = np.meshgrid(cff_range, cff_range)
    
    reh, ree = grid
    reh = reh.flatten()
    ree = ree.flatten()

    f1 = find_f_2D(reh, ree, df, angle1)
    f2 = find_f_2D(reh, ree, df, angle2)

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
    plt.savefig(f'Work/Aaryan/2D Surface Plots/ReH_ReE_lossPlot_{angle1}_{angle2}.png')
    plt.show()


create_folders('2D Surface Plots')
create_folders('3D Surface Plots')
create_2D_lossPlot(kinematicDf, 7.5, 187.5)


 

