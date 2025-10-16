import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Global variables for file paths
RESULTS_CSV_1 = 'results.csv'  # Path to first results.csv file
RESULTS_CSV_2 = '../../9_29_best_model_200_replicas/sets_1-20/results.csv'  # Path to second results.csv file
OUTPUT_DIR = 'Grid_Plots'

def create_folders(folder_name):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")

def create_cff_res_division_plot(output_dir=OUTPUT_DIR):
    """
    Create stacked division plots for ReH, ReHt, ReE, and DVCS using
    CFF_res columns from two different results.csv files.
    Division = CFF_res_file1 / CFF_res_file2
    """
    create_folders(output_dir)

    # Check if both files exist
    if not os.path.exists(RESULTS_CSV_1):
        print(f"Results CSV 1 not found: {RESULTS_CSV_1}")
        return
    if not os.path.exists(RESULTS_CSV_2):
        print(f"Results CSV 2 not found: {RESULTS_CSV_2}")
        return

    # Load both dataframes
    df1 = pd.read_csv(RESULTS_CSV_1)
    df2 = pd.read_csv(RESULTS_CSV_2)
    
    # Get unique sets (assuming both files have the same sets)
    unique_sets = sorted(df1['set'].unique())
    
    # Map CFF base names to titles
    cff_labels = ['ReH', 'ReHt', 'ReE', 'dvcs']
    cff_titles = [r"$\Re(H)_{res}$ Ratio", r"$\Re(\tilde{H})_{res}$ Ratio", r"$\Re(E)_{res}$ Ratio", r"$|T_{DVCS}|^2_{res}$ Ratio"]

    fig, axes = plt.subplots(len(cff_labels), 1, figsize=(12, 14), sharex=True)

    for i, (cff, title) in enumerate(zip(cff_labels, cff_titles)):
        ax = axes[i]

        # Get the residual columns from both dataframes
        res_col = f"{cff}_res"

        if res_col not in df1.columns or res_col not in df2.columns:
            print(f"Missing expected column {res_col} in one or both results.csv files")
            continue

        # Group by set to preserve order
        grouped1 = df1.groupby("set").first()  # take first row per set
        grouped2 = df2.groupby("set").first()  # take first row per set
        
        res1 = grouped1[res_col].reindex(unique_sets)
        res2 = grouped2[res_col].reindex(unique_sets)

        # Calculate division, handling division by zero
        division = np.divide(res1, res2, out=np.zeros_like(res1), where=res2!=0)
        
        # Plot the division
        ax.plot(unique_sets, division, 'o', markersize=4)
        ax.set_ylabel(title)
        ax.set_ylim(0, 2)  # Set y-axis scale to 0-2
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal ratio line')

        # Hide x-axis tick labels for all but the last subplot
        if i < len(cff_labels) - 1:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel("Set")

    plt.suptitle(r"$CFF_{res}$ Ratio = $\frac{CFF_{res,file1}}{CFF_{res,file2}}$", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = os.path.join(output_dir, 'CFF_Res_Division_Stacked.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"CFF residual division stacked plot saved: {output_path}")

def create_cff_std_division_plot(output_dir=OUTPUT_DIR):
    """
    Create stacked division plots for ReH, ReHt, ReE, and DVCS using
    CFF_std columns from two different results.csv files.
    Division = CFF_std_file1 / CFF_std_file2
    """
    create_folders(output_dir)

    # Check if both files exist
    if not os.path.exists(RESULTS_CSV_1):
        print(f"Results CSV 1 not found: {RESULTS_CSV_1}")
        return
    if not os.path.exists(RESULTS_CSV_2):
        print(f"Results CSV 2 not found: {RESULTS_CSV_2}")
        return

    # Load both dataframes
    df1 = pd.read_csv(RESULTS_CSV_1)
    df2 = pd.read_csv(RESULTS_CSV_2)
    
    # Get unique sets (assuming both files have the same sets)
    unique_sets = sorted(df1['set'].unique())
    
    # Map CFF base names to titles
    cff_labels = ['ReH', 'ReHt', 'ReE', 'dvcs']
    cff_titles = [r"$\Re(H)_{std}$ Ratio", r"$\Re(\tilde{H})_{std}$ Ratio", r"$\Re(E)_{std}$ Ratio", r"$|T_{DVCS}|^2_{std}$ Ratio"]

    fig, axes = plt.subplots(len(cff_labels), 1, figsize=(12, 14), sharex=True)

    for i, (cff, title) in enumerate(zip(cff_labels, cff_titles)):
        ax = axes[i]

        # Get the std columns from both dataframes
        std_col = f"{cff}_std"

        if std_col not in df1.columns or std_col not in df2.columns:
            print(f"Missing expected column {std_col} in one or both results.csv files")
            continue

        # Group by set to preserve order
        grouped1 = df1.groupby("set").first()  # take first row per set
        grouped2 = df2.groupby("set").first()  # take first row per set
        
        std1 = grouped1[std_col].reindex(unique_sets)
        std2 = grouped2[std_col].reindex(unique_sets)

        # Calculate division, handling division by zero
        division = np.divide(std1, std2, out=np.zeros_like(std1), where=std2!=0)
        
        # Plot the division
        ax.plot(unique_sets, division, 'o', markersize=4)
        ax.set_ylabel(title)
        ax.set_ylim(0, 2)  # Set y-axis scale to 0-2
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal ratio line')

        # Hide x-axis tick labels for all but the last subplot
        if i < len(cff_labels) - 1:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel("Set")

    plt.suptitle(r"$CFF_{std}$ Ratio = $\frac{CFF_{std,file1}}{CFF_{std,file2}}$", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = os.path.join(output_dir, 'CFF_Std_Division_Stacked.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"CFF std division stacked plot saved: {output_path}")

def count_points_below_one(output_dir=OUTPUT_DIR):
    """
    Count the number of points below 1 for both res and std divisions
    and write results to a text file.
    """
    create_folders(output_dir)
    
    # Check if both files exist
    if not os.path.exists(RESULTS_CSV_1):
        print(f"Results CSV 1 not found: {RESULTS_CSV_1}")
        return
    if not os.path.exists(RESULTS_CSV_2):
        print(f"Results CSV 2 not found: {RESULTS_CSV_2}")
        return

    # Load both dataframes
    df1 = pd.read_csv(RESULTS_CSV_1)
    df2 = pd.read_csv(RESULTS_CSV_2)
    
    # Get unique sets
    unique_sets = sorted(df1['set'].unique())
    
    # CFF labels
    cff_labels = ['ReH', 'ReHt', 'ReE', 'dvcs']
    
    # Initialize counters
    res_counts = {}
    std_counts = {}
    
    # Count points below 1 for residuals
    for cff in cff_labels:
        res_col = f"{cff}_res"
        if res_col in df1.columns and res_col in df2.columns:
            grouped1 = df1.groupby("set").first()
            grouped2 = df2.groupby("set").first()
            
            res1 = grouped1[res_col].reindex(unique_sets)
            res2 = grouped2[res_col].reindex(unique_sets)
            
            division = np.divide(res1, res2, out=np.zeros_like(res1), where=res2!=0)
            count_below_one = np.sum(division < 1)
            res_counts[cff] = count_below_one
    
    # Count points below 1 for std
    for cff in cff_labels:
        std_col = f"{cff}_std"
        if std_col in df1.columns and std_col in df2.columns:
            grouped1 = df1.groupby("set").first()
            grouped2 = df2.groupby("set").first()
            
            std1 = grouped1[std_col].reindex(unique_sets)
            std2 = grouped2[std_col].reindex(unique_sets)
            
            division = np.divide(std1, std2, out=np.zeros_like(std1), where=std2!=0)
            count_below_one = np.sum(division < 1)
            std_counts[cff] = count_below_one
    
    # Write results to text file
    output_path = os.path.join(output_dir, 'points_below_one_count.txt')
    with open(output_path, 'w') as f:
        f.write("Points Below 1 Count\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"File 1: {RESULTS_CSV_1}\n")
        f.write(f"File 2: {RESULTS_CSV_2}\n")
        f.write(f"Total sets: {len(unique_sets)}\n\n")
        
        f.write("CFF Residual Ratios (CFF_res_file1 / CFF_res_file2):\n")
        f.write("-" * 50 + "\n")
        for cff in cff_labels:
            if cff in res_counts:
                f.write(f"{cff}: {res_counts[cff]} points below 1\n")
        
        f.write("\nCFF Std Ratios (CFF_std_file1 / CFF_std_file2):\n")
        f.write("-" * 50 + "\n")
        for cff in cff_labels:
            if cff in std_counts:
                f.write(f"{cff}: {std_counts[cff]} points below 1\n")
    
    print(f"Points below 1 count saved: {output_path}")
    
    # Print summary to console
    print("\nSummary of points below 1:")
    print("CFF Residual Ratios:")
    for cff in cff_labels:
        if cff in res_counts:
            print(f"  {cff}: {res_counts[cff]} points")
    print("CFF Std Ratios:")
    for cff in cff_labels:
        if cff in std_counts:
            print(f"  {cff}: {std_counts[cff]} points")

def main():
    """
    Main function to generate both comparison plots and count points below 1.
    """
    print("Generating CFF comparison plots...")
    
    # Create CFF residual division plot
    print("Creating CFF residual division plot...")
    create_cff_res_division_plot()
    
    # Create CFF std division plot
    print("Creating CFF std division plot...")
    create_cff_std_division_plot()
    
    # Count points below 1
    print("Counting points below 1...")
    count_points_below_one()
    
    print("All comparison plots and count analysis have been generated successfully!")

if __name__ == "__main__":
    main()
