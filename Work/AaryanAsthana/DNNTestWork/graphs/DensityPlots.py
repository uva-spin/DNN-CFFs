import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Load the data
data_file = 'pdata.csv'
data = pd.read_csv(data_file)

# Define the specific sets of interest
#specific_sets = [102, 103, 108, 109, 110, 111, 114, 115, 121, 122, 123, 84, 96, 97, 98, 99]
specific_sets = list(range(100, 150))

# Create a new column to label specific vs. other sets
data['Group'] = data['set'].apply(lambda x: 'Specific Sets' if x in specific_sets else 'Other Sets')

# Filter data to include only specific sets
specific_data = data[data['Group'] == 'Specific Sets']

# Variables to compare
variables = ['x_b', 'k', 't', 'QQ']

# Create a PDF file to save the figures
with PdfPages('output_plots.pdf') as pdf:
    # First Figure: Density Plots
    plt.figure(figsize=(16, 10))
    for i, var in enumerate(variables, 1):
        plt.subplot(2, 2, i)  # 2x2 grid
        sns.kdeplot(data=data[data['Group'] == 'Specific Sets'][var], label='Specific Sets', shade=True)
        sns.kdeplot(data=data[data['Group'] == 'Other Sets'][var], label='Other Sets', shade=True)
        plt.title(f'Density Plot of {var} for Specific vs. Other Sets')
        plt.xlabel(var)
        plt.ylabel('Density')
        plt.legend()
    plt.tight_layout()
    pdf.savefig()  # Save the density plot figure to the PDF
    plt.close()

    # Second Figure: Scatter Plots for Specific Sets only
    plt.figure(figsize=(16, 10))
    for i, var in enumerate(variables, 1):
        plt.subplot(2, 2, i)  # 2x2 grid
        sns.scatterplot(data=specific_data, x='set', y=var, hue='Group', style='Group', s=50)
        plt.title(f'Scatter Plot of {var} by Set (Specific Sets Only)')
        plt.xlabel('Set')
        plt.ylabel(var)
        plt.legend()
    plt.tight_layout()
    pdf.savefig()  # Save the scatter plot figure to the PDF
    plt.close()

print("Figures have been saved as 'output_plots.pdf'")