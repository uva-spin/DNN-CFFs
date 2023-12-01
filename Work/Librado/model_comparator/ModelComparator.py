import tensorflow as tf
import pandas as pd
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from BHDVCS_tf_modified import TotalFLayer

def load_model_and_intermediate_model(model_path):
    full_model = tf.keras.models.load_model(model_path, custom_objects={'TotalFLayer': TotalFLayer})

    intermediate_layer_model = tf.keras.Model(inputs=full_model.input,
                                              outputs=full_model.get_layer('cff_output_layer').output)
    return full_model, intermediate_layer_model

def calculate_accuracy(predicted_values, mean_predicted):
    n = len(predicted_values)
    return np.sum((predicted_values - mean_predicted) ** 2) / n

def calculate_precision(true_values, mean_predicted):
    return 1 - np.abs((true_values - mean_predicted) / true_values)
    
def predict_cffs_and_f(intermediate_model, full_model, inputs):
    cffs = intermediate_model.predict(inputs)
    f_values = full_model.predict(inputs)
    return cffs, f_values

def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully")
    else:
        print(f"Folder '{folder_name}' already exists")

def save_predictions_to_csv(predictions, model_name, folder_name='DNNvalues'):
    create_folder_if_not_exists(folder_name)
    df = pd.DataFrame(predictions, columns=['F', 'ReH', 'ReE', 'ReHt', 'dvcs'])
    csv_filename = os.path.join(folder_name, f'{model_name}_predictions.csv')
    df.to_csv(csv_filename, index=False)
    print(f"Predictions saved to {csv_filename}")

def plot_with_error_bars(actual_values, mean_predictions, std_predictions, title, filename, folder_name, dot_size=2, x_range=None):
    fig, ax = plt.subplots()
    ax.errorbar(range(len(mean_predictions)), mean_predictions, yerr=std_predictions, fmt='o', color='red', alpha=0.5, label='Predicted', markersize=dot_size)
    ax.scatter(range(len(actual_values)), actual_values, color='blue', s=dot_size, label='Actual')
    if x_range:
        ax.set_xlim(x_range)
    ax.set_title(title)
    ax.set_xlabel('Data Points')
    ax.set_ylabel('Values')
    ax.legend()
    plot_figure(fig, filename, folder_name)

def plot_figure(fig, filename, folder_name):
    plt.savefig(os.path.join(folder_name, filename))
    plt.close(fig)

def plot_2d(actual_values, predicted_values, title, filename, folder_name, dot_size=2, x_range=None):
    fig, ax = plt.subplots()
    ax.scatter(range(len(actual_values)), actual_values, label='Actual', color='blue', s=dot_size)
    ax.scatter(range(len(predicted_values)), predicted_values, label='Predicted', color='red', alpha=0.5, s=dot_size)
    if x_range:
        ax.set_xlim(x_range)
    ax.set_title(title)
    ax.set_xlabel('Data Points')
    ax.set_ylabel('Values')
    ax.legend()
    plot_figure(fig, filename, folder_name)

def plot_3d(x, y, z, title, filename, folder_name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')
    ax.set_title(title)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plot_figure(fig, filename, folder_name)


def main():
    create_folder_if_not_exists('DNN2Dimages')
    create_folder_if_not_exists('DNN3Dimages')
    create_folder_if_not_exists('DNNvalues')
    prediction_file = 'PseudoData_from_the_Basic_Model.csv'
    prediction_data = pd.read_csv(prediction_file)
    prediction_inputs = prediction_data[['QQ', 'x_b', 't', 'phi_x', 'k']].to_numpy()

    model_folder = 'DNNmodels'
    model_paths = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if f.endswith('.h5')]

    all_f_predictions = []
    all_cff_predictions = []
    f_accuracy_values = []
    f_precision_values = []
    cff_accuracy_values = {cff_name: [] for cff_name in ['ReH', 'ReE', 'ReHt', 'dvcs']}
    cff_precision_values = {cff_name: [] for cff_name in ['ReH', 'ReE', 'ReHt', 'dvcs']}


    for model_path in model_paths:
        full_model, intermediate_model = load_model_and_intermediate_model(model_path)
        cffs, f_values = predict_cffs_and_f(intermediate_model, full_model, prediction_inputs)
        all_f_predictions.append(f_values)
        all_cff_predictions.append(cffs)
        model_predictions = np.hstack((f_values, cffs))
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        save_predictions_to_csv(model_predictions, model_name)

    all_f_predictions_array = np.array(all_f_predictions)
    all_cff_predictions_array = np.array(all_cff_predictions)

    actual_F = prediction_data['F'].values
    predicted_F = all_f_predictions_array[:, :, 0].mean(axis=0)
    plot_2d(actual_F, predicted_F, 'Actual vs Predicted F', 'actual_vs_predicted_F.png', 'DNN2Dimages')

    mean_f_predictions = all_f_predictions_array[:, :, 0].mean(axis=0)
    std_f_predictions = all_f_predictions_array[:, :, 0].std(axis=0)
    plot_with_error_bars(actual_F, mean_f_predictions, std_f_predictions, 'Actual vs Predicted F', 'actual_vs_predicted_F_error_bars.png', 'DNN2Dimages')

    f_accuracy = calculate_accuracy(predicted_F, mean_f_predictions)
    f_precision = calculate_precision(actual_F, mean_f_predictions)
    f_accuracy_values.append(f_accuracy)
    f_precision_values.append(f_precision)

    for i, cff_name in enumerate(['ReH', 'ReE', 'ReHt', 'dvcs']):
        actual_cff = prediction_data[cff_name].values
        predicted_cff = all_cff_predictions_array[:, :, i].mean(axis=0)
        cff_accuracy = calculate_accuracy(predicted_cff, predicted_cff.mean())
        cff_precision = calculate_precision(actual_cff, predicted_cff.mean())
        cff_accuracy_values[cff_name].append(cff_accuracy)
        cff_precision_values[cff_name].append(cff_precision)
        
        
    avg_f_accuracy = np.mean(f_accuracy_values)
    avg_f_precision = np.mean(f_precision_values)
    print(f"\nAverage Accuracy for F: {avg_f_accuracy}")
    print(f"Average Precision for F: {avg_f_precision}")

    for cff_name in ['ReH', 'ReE', 'ReHt', 'dvcs']:
        avg_cff_accuracy = np.mean(cff_accuracy_values[cff_name])
        avg_cff_precision = np.mean(cff_precision_values[cff_name])
        print(f"\nAverage Accuracy for {cff_name}: {avg_cff_accuracy}")
        print(f"Average Precision for {cff_name}: {avg_cff_precision}")
        
    combined_predictions = np.hstack((all_f_predictions_array.mean(axis=0), all_cff_predictions_array.mean(axis=0)))
    column_names = ['F', 'ReH', 'ReE', 'ReHt', 'dvcs']
    for (var1, var2) in itertools.combinations(column_names, 2):
        x_values = range(len(prediction_data))
        y_values = combined_predictions[:, column_names.index(var1)]
        z_values = combined_predictions[:, column_names.index(var2)]
        plot_3d(x_values, y_values, z_values, f'3D Plot of {var1} vs {var2}', f'{var1}_vs_{var2}_3d.png', 'DNN3Dimages')

if __name__ == "__main__":
    main()