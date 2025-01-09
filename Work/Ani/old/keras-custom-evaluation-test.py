import pandas as pd
import numpy as np
import tensorflow as tf
from BHDVCS_tf_modified import TotalFLayer

def calculate_statistics(model_path, data_file):
    # Load the model
    custom_objects = {'TotalFLayer': TotalFLayer}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    intermediate_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('cff_output_layer').output)
    
    # Load the data and select specific rows (excluding the header and starting from the second row)
    df = pd.read_csv(data_file)
    evaluation_df = df.iloc[1:25]  # Select rows 2 to 38 (1:38 considering zero-indexing)

    # Prepare the input features and actual CFF values from the selected rows
    inputs = evaluation_df[['k', 'QQ', 'x_b', 't', 'phi_x']].to_numpy()
    actual_cffs = evaluation_df[['ReH', 'ReE', 'ReHt', 'dvcs']].to_numpy()

    # Predict CFFs
    predicted_cffs = intermediate_model.predict(inputs)

    # Calculate statistics
    cff_names = ['ReH', 'ReE', 'ReHt', 'dvcs']
    results = {}
    for i, cff_name in enumerate(cff_names):
        mean_predicted = np.mean(predicted_cffs[:, i])
        mean_actual = np.mean(actual_cffs[:, i])
        std_dev = np.std(predicted_cffs[:, i])
        percent_error = np.abs((mean_predicted - mean_actual) / mean_actual) * 100

        results[cff_name] = {
            'Mean Predicted': mean_predicted,
            'Mean Actual': mean_actual,
            'Standard Deviation': std_dev,
            'Percent Error': percent_error
        }
    
    return results

# Example usage
model_path = 'model0.h5'
data_file = 'PseudoData_from_the_Basic_Model_for_JLab_Kinematics.csv'
statistics = calculate_statistics(model_path, data_file)

# Print the results
for cff_name, stats in statistics.items():
    print(f"{cff_name} Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    print()  # Add an empty line for better readability

