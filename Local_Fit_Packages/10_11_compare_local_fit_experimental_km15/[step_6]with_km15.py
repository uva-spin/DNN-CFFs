import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import norm
from global_fit.experimental_data_user_inputs import *
import glob

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")


def create_cff_prediction_grid_plot(output_dir='Grid_Plots', selected_sets=[24, 25, 30, 34, 35, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 58, 59, 60, 64]):
    """
    Create stacked residual plots for ReH, ReHt, ReE, and DVCS using
    DNN_projections.csv (blue, fit to experimental data) and results.csv (green, fit to KM15 pseudodata),
    overlaid with KM15 true values (red).
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Ensure directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dnn_csv = 'global_fit/DNN_projections.csv'
    results_csv = 'best_21_65/results.csv'

    if not os.path.exists(dnn_csv):
        print(f"Results CSV not found: {dnn_csv}")
        return
    if not os.path.exists(results_csv):
        print(f"Overlay results CSV not found: {results_csv}")
        return

    df_dnn = pd.read_csv(dnn_csv)
    df_results = pd.read_csv(results_csv)
    unique_sets = sorted(df_dnn['set'].unique())
    if selected_sets is not None:
        selected_sets = [s for s in selected_sets if s in unique_sets]
        if not selected_sets:
            print("No valid kinematic sets found in provided list.")
            return
        unique_sets = selected_sets
    print(unique_sets)
    cff_labels = ['ReH', 'ReHt', 'ReE', 'dvcs']
    cff_titles = [
        r"$\Re(H)$", r"$\Re(\tilde{H})$", r"$\Re(E)$", r"$|T_{DVCS}|^2$"
    ]

    fig, axes = plt.subplots(len(cff_labels), 1, figsize=(12, 14), sharex=True)

    for i, (cff, title) in enumerate(zip(cff_labels, cff_titles)):
        ax = axes[i]

        # Blue: DNN fit to experimental data
        res_col = f"{cff}_pred"
        std_col = f"{cff}_std"
        if res_col not in df_dnn.columns or std_col not in df_dnn.columns:
            print(f"Missing expected columns {res_col} or {std_col} in DNN file")
            continue

        grouped_dnn = df_dnn.groupby("set").first()
        residuals = grouped_dnn[res_col].reindex(unique_sets)
        errors = grouped_dnn[std_col].reindex(unique_sets)
        ax.errorbar(unique_sets, residuals, yerr=errors, fmt='o', capsize=3,
                    color='blue', label='DNN (fit to experimental data)')

        # Green: DNN fit to KM15 pseudodata
        true_col = f"{cff}"
        pred_col = f"{cff}_pred"
        std_col_r = f"{cff}_std"
        if all(c in df_results.columns for c in [true_col, pred_col, std_col_r]):
            grouped_res = df_results.groupby("set").first()
            true_vals = grouped_res[true_col].reindex(unique_sets)
            pred_vals = grouped_res[pred_col].reindex(unique_sets)
            std_vals = grouped_res[std_col_r].reindex(unique_sets)

            ax.errorbar(unique_sets, pred_vals, yerr=std_vals, fmt='o', capsize=3,
                        color='green', label='DNN (fit to KM15 pseudodata)')

            # Red: KM15 ground truth
            ax.scatter(unique_sets, true_vals, color='red', s=25, zorder=5, label='KM15')

        ax.set_ylabel(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=9)

        if cff == 'dvcs':
            ax.set_ylim(-0.02, 0.02)
        else:
            ax.set_ylim(-3, 3)

        if i < len(cff_labels) - 1:
            ax.tick_params(labelbottom=False)

        if i == 0:
            ax.legend(fontsize=9, loc='upper right')

    axes[-1].set_xlabel("Set", fontsize=11)

    plt.suptitle(
        "CFF Predictions: KM15 (red), DNN fit to experimental data (blue), DNN fit to KM15 pseudodata (green)",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = os.path.join(output_dir, 'CFF_Predictions_Overlay.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"CFF prediction overlay plot saved: {output_path}")



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
    #create_f_vs_phi_grid_plots()
    
    print("All grid plots have been generated successfully!")

if __name__ == "__main__":
    main()


