import os
import pandas as pd
import matplotlib.pyplot as plt
from user_inputs import *

def generate_mean_std_plots(df, cff_labels, output_dir, sets_per_plot=20, model_index=0):
    """
    Generates and saves plots for each CFF showing residual and std deviation for a given model.
    """
    os.makedirs(output_dir, exist_ok=True)

    for cff in cff_labels:
        res_col = f'{cff}_res'
        std_col = f'{cff}_std'
        sets = df['set']
        residuals = df[res_col]
        stds = df[std_col]

        for chunk_start in range(0, len(sets), sets_per_plot):
            chunk_end = min(chunk_start + sets_per_plot, len(sets))
            chunk_sets = sets[chunk_start:chunk_end]
            chunk_residuals = residuals[chunk_start:chunk_end]
            chunk_stds = stds[chunk_start:chunk_end]

            plt.figure(figsize=(10, 6))
            for i, (set_num, res, std) in enumerate(zip(chunk_sets, chunk_residuals, chunk_stds)):
                plt.plot([i + 1, i + 1], [res - std, res + std], color='blue', linewidth=2)
                plt.scatter(i + 1, res, color='blue', zorder=5)

            plt.title(f'TopModel {model_index} | {cff} Residual ± Std Dev (Sets {chunk_start + 1}-{chunk_end})')
            plt.xlabel('Kinematic Set Index')
            plt.ylabel(f'{cff} Residual')
            plt.xticks(range(1, chunk_end - chunk_start + 1), chunk_sets)
            plt.grid(True)

            filename = f'{cff}_Residual_Std_TopModel_{model_index}_Sets_{chunk_start + 1}_to_{chunk_end}.png'
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()
            print(f"✅ Saved: {filename}")

def process_all_models():
    cff_labels = ['ReH', 'ReE', 'ReHt', 'dvcs']
    sets_per_plot = 26  # Customize per your preference
    output_base = 'CFF_Mean_Deviation_Plots'

    for model_index in range(10):
        summary_csv = f'Summary_of_CFFs_TopModel_{model_index}.csv'
        if not os.path.exists(summary_csv):
            print(f"❌ File not found: {summary_csv}")
            continue

        print(f"\n📊 Generating plots for TopModel {model_index}")
        df = pd.read_csv(summary_csv)
        output_dir = os.path.join(output_base, f'TopModel_{model_index}')
        generate_mean_std_plots(df, cff_labels, output_dir, sets_per_plot, model_index)

    print("\n✅ All model plots generated.")

if __name__ == "__main__":
    process_all_models()
