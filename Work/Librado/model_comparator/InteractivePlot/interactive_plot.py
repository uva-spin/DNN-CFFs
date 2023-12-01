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

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_3d_interactive.py <y_column> <z_column>")
        print("Columns: 'F', 'ReH', 'ReE', 'ReHt', 'dvcs'")
    else:
        y_col = sys.argv[1]
        z_col = sys.argv[2]

        valid_columns = ['F', 'ReH', 'ReE', 'ReHt', 'dvcs']
        if y_col not in valid_columns or z_col not in valid_columns:
            print("Invalid columns specified. Please choose from 'F', 'ReH', 'ReE', 'ReHt', 'dvcs'.")
        else:
            try:
                avg_csv = average_predictions('../DNNvalues')
                create_interactive_3d_plot(avg_csv, y_col, z_col)
            except Exception as e:
                print(str(e))
