import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import norm

def remove_duplicate_columns(df):
    """
    Removes duplicate columns from a pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: The DataFrame with duplicate columns removed.
    """

    df = df.loc[:, ~df.columns.duplicated()]
    return df

# Read the CSV file
df = pd.read_csv('evaluation.csv')

# Remove duplicate columns
df = remove_duplicate_columns(df)

# Save the modified DataFrame back to CSV (optional)
df.to_csv('evaluation.csv', index=False)

def generate_f_vs_phi_plot_from_csv(csv_path, output_dir='Comparison_Plots'):
    """
    Generate an F vs phi_x plot using data from the CSV file.

    Parameters:
    - csv_path: Path to the CSV file containing the F vs phi_x data.
    - output_dir: Directory where the generated plot will be saved.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data from CSV
    df = pd.read_csv(csv_path)
    
    # Extract data from the CSV
    phi_x_values = df['phi_x']
    real_F_values = df['Real F']
    mean_f_predictions = df['Mean F Prediction']
    std_f_predictions = df['Std Dev Prediction']
    
    # Generate the F vs phi_x plot
    plt.figure(figsize=(10, 6))
    plt.scatter(phi_x_values, real_F_values, color='red', label='Real F', zorder=5)
    plt.errorbar(phi_x_values, real_F_values, yerr=std_f_predictions, fmt='o', color='red', ecolor='red', capsize=5, label='Real F Error', zorder=6)
    plt.plot(phi_x_values, mean_f_predictions, color='blue', label='Mean F Prediction')
    plt.fill_between(phi_x_values, mean_f_predictions - std_f_predictions, mean_f_predictions + std_f_predictions, color='blue', alpha=0.2, label='±1 Std Dev')
    
    # Set plot labels and title
    plt.xlabel('phi_x')
    plt.ylabel('F')
    plt.title('F vs phi_x with Error Bars and Error Bands')
    plt.grid(True)
    plt.legend()
    
    # Define the output file path
    kinematic_set = os.path.basename(csv_path).split('_')[-1].replace('.csv', '')
    output_png_path = os.path.join(output_dir, f'F_vs_phi_x_Plot_Set_{kinematic_set}.png')
    
    # Save the plot as a PNG file
    plt.tight_layout()
    plt.savefig(output_png_path)
    plt.close()
    print(f"F vs phi_x plot saved: {output_png_path}")

    #Generating the plot for the kinematic sets 



def generate_mean_std_plots(df, cff_labels, output_dir='CFF_Mean_Deviation_Plots', sets_per_plot=20):
    """
    Generates and saves plots for each CFF showing the residual (true - predicted) and standard deviation.
    Each plot will contain up to a specified number of kinematic sets, and new plots will be generated 
    if the number of kinematic sets exceeds the limit.

    Parameters:
    - df: DataFrame containing the data for CFFs.
    - cff_labels: List of CFF labels (e.g., ['ReH', 'ReE', 'ReHt', 'dvcs']).
    - output_dir: Directory to save the plots.
    - sets_per_plot: Number of kinematic sets per plot.
    """

    for cff in cff_labels:
        res_col = f'{cff}_res'
        std_col = f'{cff}_std'
        sets = df['set']
        residuals = df[res_col]
        stds = df[std_col]

        # Break the sets into chunks to handle the sets_per_plot limit
        for chunk_start in range(0, len(sets), sets_per_plot):
            chunk_end = min(chunk_start + sets_per_plot, len(sets))
            chunk_sets = sets[chunk_start:chunk_end]
            chunk_residuals = residuals[chunk_start:chunk_end]
            chunk_stds = stds[chunk_start:chunk_end]

            plt.figure(figsize=(10, 6))

            for i, (set_num, res, std) in enumerate(zip(chunk_sets, chunk_residuals, chunk_stds)):
                # Plot the mean and standard deviation for each kinematic set in the chunk
                plt.plot([i + 1, i + 1], [res - std, res + std], color='blue', linewidth=2)  # Std bounds
                plt.scatter(i + 1, res, color='blue', zorder=5)  # Mean as a dot

            # Set plot labels and title
            plt.title(f'{cff} Residual with Standard Deviation (Sets {chunk_start + 1}-{chunk_end})')
            plt.xlabel('Kinematic Set')
            plt.ylabel(f'{cff} Residual')
            plt.xticks(range(1, chunk_end - chunk_start + 1), chunk_sets)  # Set x-tick labels to the kinematic set numbers
            plt.grid(True)

            # Save the plot for the current chunk of sets
            plot_path = os.path.join(output_dir, f'{cff}_Residual_Std_Plot_Sets_{chunk_start + 1}_to_{chunk_end}.png')
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

def generate_plots_from_csv(csv_path, output_dir='Comparison_Plots'):
    """
    Generate plots (histograms with mean and standard deviation) from the CSV data.
    
    Parameters:
    - csv_path: Path to the CSV file containing the predictions.
    - output_dir: Directory to save the generated plots.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data from CSV
    df = pd.read_csv(csv_path)

    # Extract unique kinematic set and CFF labels
    kinematic_set = df['set'].iloc[0]
    cff_labels = df['cff_label'].unique()

    # Create a figure with subplots for each CFF
    plt.figure(figsize=(15, 10))

    for i, cff_label in enumerate(cff_labels):
        # Filter the data for the current CFF
        cff_data = df[df['cff_label'] == cff_label]

        # Extract prediction, mean, and std deviation
        predictions = cff_data['prediction']
        mean_value = cff_data['mean_value'].iloc[0]
        std_deviation = cff_data['std_deviation'].iloc[0]
        true_value = cff_data['true_value'].iloc[0]

        # Create a histogram for the predictions
        plt.subplot(2, 2, i + 1)
        plt.hist(predictions, bins=20, edgecolor='black', alpha=0.7, color='lightblue')

        # Plot vertical lines for true value, mean, and ±1 standard deviation
        plt.axvline(x=true_value, color='red', linestyle='--', label='True Value')
        plt.axvline(x=mean_value, color='blue', linestyle='--', label='Mean')
        plt.axvline(x=mean_value - std_deviation, color='green', linestyle='--', label='1-sigma')
        plt.axvline(x=mean_value + std_deviation, color='green', linestyle='--')

        # Fit a Gaussian curve for visualization
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mean_value, std_deviation)
        plt.plot(x, p * len(predictions) * (xmax - xmin) / 20, 'k', linewidth=2)

        # Set plot titles and labels
        plt.title(f'Set {kinematic_set}: {cff_label} Histogram\nMean: {mean_value:.4f}, Std Dev: {std_deviation:.4f}')
        plt.xlabel(cff_label)
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)

    # Save the figure as a PDF file
    output_pdf_path = os.path.join(output_dir, f'CFFs_Histograms_Set_{kinematic_set}.pdf')
    plt.tight_layout()
    plt.savefig(output_pdf_path)
    plt.close()
    print(f"Plots saved to {output_pdf_path}")


def generate_all_plots(evaluation_csv, cff_predictions_dir, output_dir):
    """
    Generate all the required plots for each unique kinematic set found in evaluation.csv.
    
    Parameters:
    - evaluation_csv: Path to the evaluation.csv file containing the list of kinematic sets.
    - cff_predictions_dir: Directory containing the CSVs for CFF predictions.
    - output_dir: Directory to save the generated plots.
    """
    # Load the evaluation CSV
    df_eval = pd.read_csv(evaluation_csv)
    
    # Get the list of unique kinematic sets
    unique_sets = df_eval['set'].unique()
    print(f"Found {len(unique_sets)} unique kinematic sets: {unique_sets}")

    # Loop over each unique set
    for kinematic_set in unique_sets:
        print(f"Processing Kinematic Set: {kinematic_set}")

        # Define paths for the relevant CSV files
        f_vs_phi_csv = os.path.join(f'Comparison_Plots/F_vs_phi_x_Kinematic_Set_{kinematic_set}.csv')
        cff_predictions_csv = os.path.join(cff_predictions_dir, f'CFFs_Predictions_Set_{kinematic_set}.csv')
        cff_all_sets_combined_csv = os.path.join('CFFs_AllSets_Combined.csv')


        # Check if the necessary CSV files exist before proceeding
        if os.path.exists(f_vs_phi_csv):
            print(f"Generating F vs phi_x plot for Set {kinematic_set}")
            generate_f_vs_phi_plot_from_csv(f_vs_phi_csv, output_dir)
        else:
            print(f"F vs phi_x CSV not found for Set {kinematic_set}")

        if os.path.exists(cff_predictions_csv):
            print(f"Generating CFF histogram plots for Set {kinematic_set}")
            #generate_plots_from_csv(cff_predictions_csv, output_dir)
        else:
            print(f"CFF predictions CSV not found for Set {kinematic_set}")
        if os.path.exists(cff_all_sets_combined_csv):
            print("Generating line plots")
            daaframe = pd.read_csv(cff_all_sets_combined_csv)
            generate_mean_std_plots(daaframe, cff_labels=['ReH', 'ReE', 'ReHt', 'dvcs'], sets_per_plot=26)

    print("All plots have been generated.")

# Example usage
evaluation_csv = 'evaluation.csv'
cff_predictions_dir = '/scratch/qzf7nj/DNN_CFFs/part_3_100_125_architecture_64_layer/'
output_dir = 'Comparison_Plots'

generate_all_plots(evaluation_csv, cff_predictions_dir, output_dir)


