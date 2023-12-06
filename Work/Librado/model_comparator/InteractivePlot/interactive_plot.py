import plotly.express as px
import pandas as pd
import sys
import os
import numpy as np

def average_predictions(folder_path, pseudodata_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not files:
        raise Exception("No CSV files found in the specified folder.")

    all_predictions = []
    for file in files:
        df = pd.read_csv(os.path.join(folder_path, file))
        all_predictions.append(df)

    avg_df = pd.concat(all_predictions).groupby(level=0).mean()
    pseudodata_df = pd.read_csv(pseudodata_path)
    merged_df = pd.concat([pseudodata_df, avg_df], axis=1)
    merged_df.to_csv('average_predictions.csv', index=False)
    return 'average_predictions.csv'

def create_interactive_3d_plot(csv_file, x_col, y_col, z_col):
    df = pd.read_csv(csv_file)
    fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, title=f'3D Plot of {y_col} with {x_col} and {z_col}')
    fig.show()

def create_interactive_2d_plot(csv_file, x_col, y_col):
    df = pd.read_csv(csv_file)
    fig = px.scatter(df, x=x_col, y=y_col, title=f'2D Plot of {y_col} vs {x_col}')
    fig.show()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python interactive_plot.py <plot_type> <x_column> <y_column> [<z_column>]")
        sys.exit(1)

    plot_type = sys.argv[1]
    x_col = sys.argv[2]
    y_col = sys.argv[3]
    z_col = sys.argv[4] if len(sys.argv) == 5 else None

    valid_columns = ['F', 'ReH', 'ReE', 'ReHt', 'dvcs', 'QQ', 'x_b', 't', 'phi_x', 'k']
    if x_col not in valid_columns or y_col not in valid_columns or (z_col and z_col not in valid_columns):
        print("Invalid columns specified. Please choose from the specified columns.")
        sys.exit(1)

    try:
        avg_csv = average_predictions('../DNNvalues', '../PseudoData_from_the_Basic_Model.csv')
        if plot_type == '2d':
            create_interactive_2d_plot(avg_csv, x_col, y_col)
        elif plot_type == '3d':
            if not z_col:
                print("Z column required for 3D plot.")
                sys.exit(1)
            create_interactive_3d_plot(avg_csv, x_col, y_col, z_col)
        else:
            print("Invalid plot type. Choose '2d' or '3d'.")
    except Exception as e:
        print(str(e))
