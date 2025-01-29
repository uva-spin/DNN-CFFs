import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import numpy as np

# Specify the file path
data_file = 'pdata.csv'

# Read the CSV file
data = pd.read_csv(data_file)

# Group by 'set' and take the first occurrence of each unique set
unique_data = data.groupby('set').first().reset_index()

# Plot settings for consistency
def create_plot(x_data, y_data, x_label, y_label, title, save_name):
    plt.figure(figsize=(10, 6))
    plt.bar(x_data, y_data, width=0.8, color='steelblue')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

# 1. x_b vs Set
create_plot(unique_data['set'], unique_data['x_b'], 
            'Kinematic Set', 'x_b', 'Histogram of x_b vs Kinematic Set', 
            'xb_vs_set_histogram.png')

# 2. k vs Set
create_plot(unique_data['set'], unique_data['k'], 
            'Kinematic Set', 'k', 'Histogram of k vs Kinematic Set', 
            'k_vs_set_histogram.png')

# 3. t vs Set
create_plot(unique_data['set'], unique_data['t'], 
            'Kinematic Set', 't', 'Histogram of t vs Kinematic Set', 
            't_vs_set_histogram.png')

# 4. Q^2 (QQ) vs Set
create_plot(unique_data['set'], unique_data['QQ'], 
            'Kinematic Set', 'Q^2 (QQ)', 'Histogram of Q^2 vs Kinematic Set', 
            'QQ_vs_set_histogram.png')

# 5. x_b vs QQ
plt.figure(figsize=(8, 6))
sets_to_label = [1, 2, 3]
hb = plt.hexbin(data['x_b'], data['QQ'], gridsize=30, cmap='plasma', mincnt=1)
cb = plt.colorbar(hb)
for idx, row in data.iterrows():
    if row['set'] in sets_to_label:
        plt.text(row['x_b'], row['QQ'], str(row['set']), color='black', fontsize=8, ha='center', va='center')
plt.title('x_b vs QQ')
plt.xlabel('x_b')
plt.ylabel('Q^2 (QQ)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. t vs QQ
plt.figure(figsize=(8, 6))
hb = plt.hexbin(data['t'], data['QQ'], gridsize=30, cmap='viridis', mincnt=14)
cb = plt.colorbar(hb)
counts = hb.get_array()
verts = hb.get_offsets()
for i, count in enumerate(counts):
    if count > 0:
        plt.text(verts[i, 0], verts[i, 1], int(count), color='black', fontsize=8, ha='center', va='center')
plt.title('t vs QQ')
plt.xlabel('t')
plt.ylabel('Q^2 (QQ)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Analyzing specific sets
specific_sets = [102, 103, 108, 109, 110, 111, 114, 115, 121, 122, 123, 84, 96, 97, 98, 99]
filtered_data = data[data['set'].isin(specific_sets)]

# Generate data subsets for saving results
xb_vs_set = filtered_data[['set', 'x_b']].drop_duplicates()
k_vs_set = filtered_data[['set', 'k']].drop_duplicates()
t_vs_set = filtered_data[['set', 't']].drop_duplicates()
qq_vs_set = filtered_data[['set', 'QQ']].drop_duplicates()
xb_vs_qq = filtered_data[['x_b', 'QQ', 'set']].drop_duplicates()
t_vs_qq = filtered_data[['t', 'QQ', 'set']].drop_duplicates()

# Compile results for easier saving or display
results_summary = {
    "x_b vs Set": xb_vs_set,
    "k vs Set": k_vs_set,
    "t vs Set": t_vs_set,
    "Q^2 (QQ) vs Set": qq_vs_set,
    "x_b vs QQ": xb_vs_qq,
    "t vs QQ": t_vs_qq
}

output_file = './specific_sets_results.txt'
with open(output_file, 'w') as file:
    for plot_name, df in results_summary.items():
        file.write(f"Results for {plot_name}:\n")
        file.write(df.to_string(index=False))
        file.write("\n\n")

# Calculate and save summary statistics
summary_stats = {
    "Mean": filtered_data[['x_b', 'k', 't', 'QQ']].mean(),
    "Median": filtered_data[['x_b', 'k', 't', 'QQ']].median(),
    "Standard Deviation": filtered_data[['x_b', 'k', 't', 'QQ']].std(),
    "Range": filtered_data[['x_b', 'k', 't', 'QQ']].apply(lambda x: x.max() - x.min())
}

summary_df = pd.DataFrame(summary_stats)

# Save summary statistics to file
summary_output_file = './specific_sets_summary_statistics.txt'
with open(summary_output_file, 'w') as file:
    file.write("Summary Statistics for Specific Sets:\n")
    file.write(summary_df.to_string())
    file.write("\n\n")

# Calculate and save correlation matrix
correlation_matrix = filtered_data[['x_b', 'k', 't', 'QQ']].corr()
with open(summary_output_file, 'a') as file:
    file.write("Correlation Matrix:\n")
    file.write(correlation_matrix.to_string())
