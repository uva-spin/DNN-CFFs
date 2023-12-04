import plotly.express as px
import pandas as pd
import sys
import os
import numpy as np

def average_predictions(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not files:
        raise Exception("No CSV files found in the specified folder.")

    all_predictions = []
    for file in files:
        df = pd.read_csv(os.path.join(folder_path, file))
        all_predictions.append(df)

    avg_df = pd.concat(all_predictions).groupby(level=0).mean()
    avg_df.to_csv('average_predictions.csv', index=False)
    return 'average_predictions.csv'

def create_interactive_3d_plot(csv_file, y_col, z_col):
    df = pd.read_csv(csv_file)
    df['Bin'] = df.index

    fig = px.scatter_3d(df, x='Bin', y=y_col, z=z_col)
    fig.show()

def create_interactive_2d_plot(csv_file, x_col, y_col):
    df = pd.read_csv(csv_file)
    if x_col == 'Bin':
        df['Bin'] = df.index
    fig = px.scatter(df, x=x_col, y=y_col)
    fig.show()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python plot_interactive.py <plot_type> <x_column> <y_column>")
        print("plot_type: '2d' or '3d'")
        print("Columns for 3D: 'F', 'ReH', 'ReE', 'ReHt', 'dvcs'")
        print("Columns for 2D: 'Bin', 'F', 'ReH', 'ReE', 'ReHt', 'dvcs'")
    else:
        plot_type = sys.argv[1]
        x_col = sys.argv[2]
        y_col = sys.argv[3]

        valid_columns = ['Bin', 'F', 'ReH', 'ReE', 'ReHt', 'dvcs']
        if x_col not in valid_columns or y_col not in valid_columns:
            print("Invalid columns specified. Please choose from 'Bin', 'F', 'ReH', 'ReE', 'ReHt', 'dvcs'.")
        else:
            try:
                avg_csv = average_predictions('../DNNvalues')
                if plot_type == '3d':
                    create_interactive_3d_plot(avg_csv, x_col, y_col)
                elif plot_type == '2d':
                    create_interactive_2d_plot(avg_csv, x_col, y_col)
                else:
                    print("Invalid plot type. Please choose '2d' or '3d'.")
            except Exception as e:
                print(str(e))
