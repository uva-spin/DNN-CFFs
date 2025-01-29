import pandas as pd

# Load the uploaded CSV file to examine its structure and contents
file_path = '/Users/aaryanasthana/Development/SpinPhysics/DNNTestWork/CSV Testing/AllJlabData_from_Zulkaida_and_Liliet.csv'
data = pd.read_csv(file_path)

# Remove rows where 'F' is 0
filtered_data = data[data['F'] != 0]

# Count the number of data points for each set
filtered_set_counts = filtered_data['#Set'].value_counts()

# Find sets with 10 or fewer data points
sets_with_few_points = filtered_set_counts[filtered_set_counts <= 10]

# Display the sets with their counts
print("Sets with 10 or fewer data points:")
print(sets_with_few_points)

# Save the filtered sets to a new CSV file if needed
filtered_data.to_csv('filtered_data.csv', index=False)