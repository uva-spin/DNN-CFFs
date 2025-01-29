import pandas as pd

# Load the data
data_file = 'pdata.csv'
data = pd.read_csv(data_file)

def get_values_for_set(data, set_number, columns):
    """
    Returns unique values for specified columns for a given set number.
    
    Parameters:
    - data: The pandas DataFrame containing the dataset.
    - set_number: The set number to search for (e.g., 102).
    - columns: List of column names to retrieve values for (e.g., ['ReH', 'ReE', 'ReHt', 'dvcs']).
    
    Returns:
    - A dictionary with unique values for each specified column in the given set number.
    """
    # Filter the data for the specified set
    set_data = data[data['set'] == set_number]
    
    # Initialize a dictionary to store results
    result = {}

    # Loop through each column and retrieve unique value
    for column in columns:
        if column in set_data.columns:
            unique_values = set_data[column].unique()   
            # Store the single unique value if there's only one, otherwise store all unique values
            result[column] = unique_values[0] if len(unique_values) == 1 else unique_values
        else:
            result[column] = f"Column '{column}' does not exist in the dataset."
    
    return result

# Example usage:
set_number = 99
columns = ['ReH', 'ReE', 'ReHt', 'dvcs']
result = get_values_for_set(data, set_number, columns)

print(f"The values for set {set_number} are: {result}")