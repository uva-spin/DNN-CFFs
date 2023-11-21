import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from BHDVCS_tf_modified import TotalFLayer

def load_model_and_intermediate_model(model_path):
    full_model = tf.keras.models.load_model(model_path, custom_objects={'TotalFLayer': TotalFLayer})

    intermediate_layer_model = tf.keras.Model(inputs=full_model.input,
                                              outputs=full_model.get_layer('cff_output_layer').output)
    return full_model, intermediate_layer_model

def predict_cffs_and_f(intermediate_model, full_model, inputs):
    cffs = intermediate_model.predict(inputs)
    f_values = full_model.predict(inputs)
    return cffs, f_values

def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")

def plot_actual_vs_predicted(actual_values, predicted_values, title, filename, dot_size=2, x_range=None):
    plt.figure()
    plt.scatter(range(len(actual_values)), actual_values, label='Actual', color='blue', s=dot_size)
    plt.scatter(range(len(predicted_values)), predicted_values, label='Predicted', color='red', alpha=0.5, s=dot_size)
    if x_range is not None:
        plt.xlim(x_range)
    plt.title(title)
    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig(os.path.join('DNNimages', filename))

def save_predictions_to_csv(predictions, model_name, folder_name='DNNvalues'):
    create_folder_if_not_exists(folder_name)
    df = pd.DataFrame(predictions, columns=['F', 'ReH', 'ReE', 'ReHt', 'dvcs'])
    csv_filename = os.path.join(folder_name, f'{model_name}_predictions.csv')
    df.to_csv(csv_filename, index=False)
    print(f"Predictions saved to {csv_filename}")

def plot_with_error_bars(actual_values, predicted_values_array, title, filename, dot_size=2, x_range=None):
    mean_predictions = np.mean(predicted_values_array, axis=0)
    std_predictions = np.std(predicted_values_array, axis=0)

    plt.figure()
    plt.errorbar(range(len(mean_predictions)), mean_predictions, yerr=std_predictions, fmt='o', color='red', alpha=0.5, label='Predicted')
    plt.scatter(range(len(actual_values)), actual_values, color='blue', s=dot_size, label='Actual')
    if x_range is not None:
        plt.xlim(x_range)
    plt.title(title)
    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig(os.path.join('DNNimages', filename))
    plt.close()  # Close the figure to free memory

def main():
    create_folder_if_not_exists('DNNimages')
    create_folder_if_not_exists('DNNvalues')
    prediction_file = 'PseudoData_from_the_Basic_Model.csv'
    prediction_data = pd.read_csv(prediction_file)
    prediction_inputs = prediction_data[['QQ', 'x_b', 't', 'phi_x', 'k']].to_numpy()

    model_folder = 'DNNmodels'
    model_paths = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if f.endswith('.h5')]

    all_f_predictions = []
    all_cff_predictions = []

    for model_path in model_paths:
        full_model, intermediate_model = load_model_and_intermediate_model(model_path)
        cffs, f_values = predict_cffs_and_f(intermediate_model, full_model, prediction_inputs)
        all_f_predictions.append(f_values)
        all_cff_predictions.append(cffs)
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        print(model_name)
        combined_predictions = np.hstack((f_values, cffs))
        save_predictions_to_csv(combined_predictions, model_name)

    all_f_predictions_array = np.array(all_f_predictions)
    all_cff_predictions_array = np.array(all_cff_predictions)

    actual_F = prediction_data['F'].values
    plot_with_error_bars(actual_F, all_f_predictions_array[:, :, 0], 'Actual vs Predicted F', 'actual_vs_predicted_F_error_bars.png')

    for i, cff_name in enumerate(['ReH', 'ReE', 'ReHt', 'dvcs']):
        actual_cff = prediction_data[cff_name].values
        plot_with_error_bars(actual_cff, all_cff_predictions_array[:, :, i], f'Actual vs Predicted {cff_name}', f'actual_vs_predicted_{cff_name}_error_bars.png')

if __name__ == "__main__":
    main()
