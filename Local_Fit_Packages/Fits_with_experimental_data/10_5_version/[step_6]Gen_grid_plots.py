import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import norm
from experimental_data_user_inputs import *
import glob

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")

def create_cff_prediction_grid_plot(output_dir='Grid_Plots'):
    """
    Create stacked residual plots for ReH, ReHt, ReE, and DVCS using
    precomputed residual (_res) and std (_std) columns in results.csv.
    """
    create_folders(output_dir)

    results_csv = 'DNN_projections.csv'
    if not os.path.exists(results_csv):
        print(f"Results CSV not found: {results_csv}")
        return

    df = pd.read_csv(results_csv)
    unique_sets = sorted(df['set'].unique())

    # Map CFF base names to titles
    cff_labels = ['ReH', 'ReHt', 'ReE', 'dvcs']
    cff_titles = [r"Res $\Re(H)$", r"Res $\Re(\tilde{H})$", r"Res $\Re(E)$", r"Res $|T_{DVCS}|^2$"]

    fig, axes = plt.subplots(len(cff_labels), 1, figsize=(12, 14), sharex=True)

    for i, (cff, title) in enumerate(zip(cff_labels, cff_titles)):
        ax = axes[i]

        # Pull the residuals and stds directly from the DataFrame
        res_col = f"{cff}_pred"
        std_col = f"{cff}_std"

        if res_col not in df.columns or std_col not in df.columns:
            print(f"Missing expected columns {res_col} or {std_col} in results.csv")
            continue

        # Group by set to preserve order
        grouped = df.groupby("set").first()  # take first row per set
        residuals = grouped[res_col].reindex(unique_sets)
        errors = grouped[std_col].reindex(unique_sets)

        ax.errorbar(unique_sets, residuals, yerr=errors, fmt='o', capsize=3)
        ax.set_ylabel(title)
        if cff == 'dvcs':
            ax.set_ylim(-0.02, 0.02)
        else:
            ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.3)

        # Hide x-axis tick labels for all but the last subplot
        if i < len(cff_labels) - 1:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel("Set")

    plt.suptitle("Predictions", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = os.path.join(output_dir, 'CFF_Predictions_Stacked.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"CFF residual stacked plot saved: {output_path}")


def create_cff_percent_diff_grid_plot(output_dir='Grid_Plots'):
    """
    Create stacked percent difference plots for ReH, ReHt, ReE, and DVCS using
    true (_true), predicted (_pred), and std (_std) columns in results.csv.
    Percent difference = (predicted - true) / true * 100
    Error bars represent percent difference of (pred ± std) vs true.
    """
    create_folders(output_dir)

    results_csv = 'DNN_projections.csv'
    if not os.path.exists(results_csv):
        print(f"Results CSV not found: {results_csv}")
        return

    df = pd.read_csv(results_csv)
    unique_sets = sorted(df['set'].unique())

    # Map CFF base names to titles
    cff_labels = ['ReH', 'ReHt', 'ReE', 'dvcs']
    cff_titles = [r"% Diff $\Re(H)$", r"% Diff $\Re(\tilde{H})$", r"% Diff $\Re(E)$", r"% Diff $|T_{DVCS}|^2$"]

    fig, axes = plt.subplots(len(cff_labels), 1, figsize=(12, 14), sharex=True)

    for i, (cff, title) in enumerate(zip(cff_labels, cff_titles)):
        ax = axes[i]

        # Pull the true values, predictions, and stds from the DataFrame
        true_col = f"{cff}"
        pred_col = f"{cff}_pred"
        std_col = f"{cff}_std"

        if true_col not in df.columns or pred_col not in df.columns or std_col not in df.columns:
            print(f"Missing expected columns {true_col}, {pred_col}, or {std_col} in results.csv")
            continue

        # Group by set to preserve order
        grouped = df.groupby("set").first()  # take first row per set
        true_vals = grouped[true_col].reindex(unique_sets)
        pred_vals = grouped[pred_col].reindex(unique_sets)
        std_vals = grouped[std_col].reindex(unique_sets)

        # Calculate percent differences
        percent_diff = ((pred_vals - true_vals) / np.abs(true_vals)) * 100
        
        # Calculate error bars for percent differences
        # Upper error: percent diff of (pred + std) vs true
        upper_pred = pred_vals + std_vals
        upper_percent_diff = ((upper_pred - true_vals) / np.abs(true_vals)) * 100
        upper_error = upper_percent_diff - percent_diff
        
        # Lower error: percent diff of (pred - std) vs true  
        lower_pred = pred_vals - std_vals
        lower_percent_diff = ((lower_pred - true_vals) / np.abs(true_vals)) * 100
        lower_error = percent_diff - lower_percent_diff

        # Create asymmetric error bars
        ax.errorbar(unique_sets, percent_diff, yerr=[lower_error, upper_error], 
                   fmt='o', capsize=3, capthick=1)
        ax.set_ylabel(title)
        
        # Set y-axis limits based on CFF type
        if cff == 'dvcs':
            ax.set_ylim(-100, 100)  # Percent differences can be large for DVCS
        else:
            ax.set_ylim(-200, 200)  # Allow for large percent differences for CFFs
        
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)  # Add zero line

        # Hide x-axis tick labels for all but the last subplot
        if i < len(cff_labels) - 1:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel("Set")

    plt.suptitle(r"Percent Difference = $\frac{CFF_{pred} - CFF_{true}}{|CFF_{true}|} \times 100$", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = os.path.join(output_dir, 'CFF_Percent_Differences_Stacked.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"CFF percent difference stacked plot saved: {output_path}")


def create_f_vs_phi_grid_plots(output_dir='Grid_Plots', plots_per_grid=20):
    """
    Create grid plots for F vs phi_x plots.
    For sets 1-20: 2 grids with 20 plots each
    For sets 180-195: 1 grid with 15 plots
    """
    create_folders(output_dir)
    
    # Determine the kinematic sets based on the current directory
    
    kinematic_sets = list(range(1, 21))
    num_grids = 2
    plots_per_grid = 20

    
    print(f"Creating {num_grids} grid(s) with up to {plots_per_grid} plots each for sets: {kinematic_sets}")
    
    for grid_num in range(num_grids):
        start_idx = grid_num * plots_per_grid
        end_idx = min(start_idx + plots_per_grid, len(kinematic_sets))
        grid_sets = kinematic_sets[start_idx:end_idx]
        
        if not grid_sets:
            continue
            
        # Calculate grid dimensions (aim for roughly square grid)
        num_plots = len(grid_sets)
        cols = int(np.ceil(np.sqrt(num_plots)))
        rows = int(np.ceil(num_plots / cols))
        
        # Create the grid
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if num_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.flatten()
        
        for i, set_num in enumerate(grid_sets):
            ax = axes[i] if num_plots > 1 else axes[0]
            
            # Load F vs phi data for this set
            csv_path = f'Comparison_Plots/F_vs_phi_x_Kinematic_Set_{set_num}.csv'
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                
                phi_x_values = df['phi_x']
                real_F_values = df['Real F']
                mean_f_predictions = df['Mean F Prediction']
                std_f_predictions = df['Std Dev Prediction']
                
                # Plot the data
                ax.scatter(phi_x_values, real_F_values, color='red', s=20, label='Real F', zorder=5)
                ax.errorbar(phi_x_values, real_F_values, yerr=std_f_predictions, fmt='o', 
                           color='red', ecolor='red', capsize=2, markersize=3, zorder=6)
                ax.plot(phi_x_values, mean_f_predictions, color='blue', linewidth=1.5, label='Mean Prediction')
                ax.fill_between(phi_x_values, mean_f_predictions - std_f_predictions, 
                               mean_f_predictions + std_f_predictions, color='blue', alpha=0.2)
                
                ax.set_title(f'Set {set_num}', fontsize=10)
                ax.set_xlabel('φ_x', fontsize=8)
                ax.set_ylabel('F', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=6)
                
                # Only show legend for the first plot
                if i == 0:
                    ax.legend(fontsize=6)
            else:
                ax.text(0.5, 0.5, f'No data for Set {set_num}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Set {set_num} (No Data)', fontsize=10)
        
        # Hide unused subplots
        for i in range(num_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'F vs φ_x Plots - Grid {grid_num + 1} (Sets {grid_sets[0]}-{grid_sets[-1]})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save the grid plot
        output_path = os.path.join(output_dir, f'F_vs_Phi_Grid_{grid_num + 1}_Sets_{grid_sets[0]}_to_{grid_sets[-1]}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"F vs phi grid {grid_num + 1} saved: {output_path}")



def main():
    """
    Main function to generate all grid plots.
    """
    print("Generating grid plots...")
    
    # Create CFF residual grid plot (all CFFs combined)
    print("Creating CFF residual grid plot...")
    create_cff_prediction_grid_plot()
    
    # Create CFF percent difference grid plot (all CFFs combined)
    #print("Creating CFF percent difference grid plot...")
    #create_cff_percent_diff_grid_plot()
    
    # Create F vs phi grid plots
    print("Creating F vs phi grid plots...")
    create_f_vs_phi_grid_plots()
    
    print("All grid plots have been generated successfully!")

if __name__ == "__main__":
    main()


