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

def create_interactive_plot(csv_file, x_col, y_col, z_col=None, cff_slices=None):
    df = pd.read_csv(csv_file)
    if cff_slices:
        # Dynamically create the query condition based on y-value
        query_conditions = []
        for cff_name, cff_range in cff_slices.items():
            if cff_name != y_col:
                query_conditions.append(f"{cff_name} >= @cff_slices['{cff_name}'][0] and {cff_name} <= @cff_slices['{cff_name}'][1]")
        query_string = " and ".join(query_conditions)
        df = df.query(query_string)

    if z_col:
        fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, title=f'3D Plot of {y_col} with {x_col} and {z_col}')
    else:
        fig = px.scatter(df, x=x_col, y=y_col, title=f'2D Plot of {y_col} vs {x_col}')
    fig.show()

def get_user_input():
    z_col = 0
    print("Do you want a 2D or 3D plot?")
    plot_type = input("Enter '2d' or '3d': ").lower()
    x_col = input("Enter the x-axis column: ")
    y_col = input("Enter the y-axis column: ")
    if plot_type == '3d':
        z_col = input("Enter the z-axis column: ")

    use_default_cff_ranges = input("Would you like to use default ranges for other CFF values (y/n)? ").lower() == 'y'
    cff_slices = {}
    if not use_default_cff_ranges:
        # Only prompt for ranges of CFFs other than the y-value
        for cff_name in ["ReH", "ReE", "ReHt"]:
            if cff_name != y_col:
                cff_slices[cff_name] = tuple(float(val) for val in input(f"Enter {cff_name} range (min, max): ").split(','))
    return plot_type, x_col, y_col, z_col, cff_slices

def main():
    try:
        plot_type, x_col, y_col, z_col, cff_slices = get_user_input()
        avg_csv = average_predictions('../DNNvalues', '../PseudoData_from_the_Basic_Model.csv')
        create_interactive_plot(avg_csv, x_col, y_col, z_col, cff_slices)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
