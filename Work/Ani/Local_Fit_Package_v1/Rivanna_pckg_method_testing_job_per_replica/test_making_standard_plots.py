import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to generate mean and standard deviation plots
def generate_mean_std_plots(df, cff_labels, output_dir='./Comparison_Plots_old', sets_per_plot=20):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for cff in cff_labels:
        pred_col = f'{cff}_pred'
        std_col = f'{cff}_std'
        sets = df['set']
        preds = df[pred_col]
        stds = df[std_col]

        # Break the sets into chunks to handle the sets_per_plot limit
        for chunk_start in range(0, len(sets), sets_per_plot):
            chunk_end = min(chunk_start + sets_per_plot, len(sets))
            chunk_sets = sets[chunk_start:chunk_end]
            chunk_preds = preds[chunk_start:chunk_end]
            chunk_stds = stds[chunk_start:chunk_end]

            plt.figure(figsize=(10, 6))

            for i, (set_num, pred, std) in enumerate(zip(chunk_sets, chunk_preds, chunk_stds)):
                plt.plot([i + 1, i + 1], [pred - std, pred + std], color='blue', linewidth=2)
                plt.scatter(i + 1, pred, color='blue', zorder=5)  # Mean as a dot

            plt.title(f'{cff} Mean with Standard Deviation (Sets {chunk_start + 1}-{chunk_end})')
            plt.xlabel('Kinematic Set')
            plt.ylabel(f'{cff}')
            plt.xticks(range(1, chunk_end - chunk_start + 1), chunk_sets)
            plt.grid(True)

            plot_path = os.path.join(output_dir, f'{cff}_Mean_Std_Plot_Sets_{chunk_start + 1}_to_{chunk_end}.png')
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

# Load the CSV file
csv_file_path = 'CFFs_AllSets_25Rows.csv'  # Path to your CSV file
df = pd.read_csv(csv_file_path)

# Example usage: generate plots for CFFs
generate_mean_std_plots(df, cff_labels=['ReH', 'ReE', 'ReHt', 'dvcs'], sets_per_plot=20)

print("Plots have been generated and saved to 'Comparison_Plots_old' folder.")
