import pandas as pd

# File path for the input CSV
input_file = '../../Sample_Codes_to_Get_Started/Example_PseudoData_Generation/Pseudo_data_Basic_Model/AllJlabData_from_Zulkaida_and_Liliet.csv'

# Read the CSV file
df = pd.read_csv(input_file)

# Filter the dataframe to include only sets 1 to 10
filtered_df = df[df['#Set'].between(1, 10)]

# Write the filtered data to a new CSV file
output_file = 'Filtered_Kinematic_Sets_1_to_10.csv'
filtered_df.to_csv(output_file, index=False)

print(f"Filtered data has been saved to {output_file}.")
