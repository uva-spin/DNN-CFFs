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

def calculate_precision(predicted_values, mean_predicted):
    n = len(predicted_values)
    return np.sum((predicted_values - mean_predicted) ** 2) / n

def calculate_accuracy(true_values, mean_predicted):
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

def plot_with_error_bars(x_values, y_values, y_errors, actual_values, title, filename, folder_name, x_label='X Axis', y_label='Y Axis', dot_size=2):
    fig, ax = plt.subplots()
    # Plotting predicted values with error bars
    ax.errorbar(x_values, y_values, yerr=y_errors, fmt='o', color='red', alpha=0.5, label='Predicted', markersize=dot_size)
    # Plotting actual values
    ax.scatter(x_values, actual_values, color='blue', s=dot_size, label='Actual')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    plot_figure(fig, filename, folder_name)

def plot_std_distribution_and_calculate_success(actual_values, predicted_values_array, title, filename, folder_name):
    mean_predictions = np.mean(predicted_values_array, axis=0)
    std_predictions = np.std(predicted_values_array, axis=0)

    # Plotting the distribution of standard deviations
    fig, ax = plt.subplots()
    ax.errorbar(range(len(mean_predictions)), mean_predictions, yerr=std_predictions, fmt='o', color='red', alpha=0.5, label='Predicted')
    ax.scatter(range(len(actual_values)), actual_values, color='blue', s=2, label='Actual')
    ax.set_title(title)
    ax.set_xlabel('Bins')
    ax.set_ylabel('Values')
    ax.legend()
    plot_figure(fig, filename, folder_name)

    # Calculate success percentage
    within_one_std = np.sum((actual_values >= (mean_predictions - std_predictions)) & (actual_values <= (mean_predictions + std_predictions)))
    success_percentage = (within_one_std / len(actual_values)) * 100
    return success_percentage


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

def plot_3d(x, y, z, title, filename, folder_name, x_label='X Axis', y_label='Y Axis', z_label='Z Axis'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)
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
    #x_values, y_values, y_errors, actual_values, title, filename, folder_name
    #plot_with_error_bars(actual_F, mean_f_predictions, std_f_predictions, 'Actual vs Predicted F', 'actual_vs_predicted_F_error_bars.png',)


#    f_accuracy = calculate_accuracy(predicted_F, mean_f_predictions)
#    f_precision = calculate_precision(actual_F, mean_f_predictions)
#    f_accuracy_values.append(f_accuracy)
#    f_precision_values.append(f_precision)

#    for i, cff_name in enumerate(['ReH', 'ReE', 'ReHt', 'dvcs']):
#        actual_cff = prediction_data[cff_name].values
#        predicted_cff = all_cff_predictions_array[:, :, i].mean(axis=0)
#        cff_accuracy = calculate_accuracy(predicted_cff, predicted_cff.mean())
#        cff_precision = calculate_precision(actual_cff, predicted_cff.mean())
#        cff_accuracy_values[cff_name].append(cff_accuracy)
#        cff_precision_values[cff_name].append(cff_precision)
        
        
#    avg_f_accuracy = np.mean(f_accuracy_values)
#    avg_f_precision = np.mean(f_precision_values)
#    print(f"\nAverage Accuracy for F: {avg_f_accuracy}")
#    print(f"Average Precision for F: {avg_f_precision}")

#    for cff_name in ['ReH', 'ReE', 'ReHt', 'dvcs']:
#        avg_cff_accuracy = np.mean(cff_accuracy_values[cff_name])
#        avg_cff_precision = np.mean(cff_precision_values[cff_name])
#        print(f"\nAverage Accuracy for {cff_name}: {avg_cff_accuracy}")
#        print(f"Average Precision for {cff_name}: {avg_cff_precision}")

        
    kinematic_vars = ['QQ', 'x_b', 't']
    cff_names = ['ReH', 'ReE', 'ReHt', 'dvcs']

    # 2D Plots for each Compton form factor against each kinematic variable and bins
    for i, cff_name in enumerate(cff_names):
        actual_cff = prediction_data[cff_name].values
        for kinematic_var in kinematic_vars:
            x_values = prediction_data[kinematic_var].values
            y_values = all_cff_predictions_array[:, :, i].mean(axis=0)
            y_errors = all_cff_predictions_array[:, :, i].std(axis=0)
            plot_with_error_bars(x_values, y_values, y_errors, actual_cff, f'{cff_name} vs {kinematic_var}', f'{cff_name}_vs_{kinematic_var}.png', 'DNN2Dimages', x_label=kinematic_var, y_label=cff_name)

        # Plotting against bins
        bins = range(len(prediction_data))
        y_values = all_cff_predictions_array[:, :, i].mean(axis=0)
        y_errors = all_cff_predictions_array[:, :, i].std(axis=0)
        plot_with_error_bars(bins, y_values, y_errors, actual_cff, f'{cff_name} vs Bins', f'{cff_name}_vs_Bins.png', 'DNN2Dimages', x_label='Bins', y_label=cff_name)

    # 3D Plots for each Compton form factor against pairs of kinematic variables
    for i, cff_name in enumerate(cff_names):
        for (var1, var2) in itertools.combinations(kinematic_vars, 2):
            x_values = prediction_data[var1].values
            y_values = prediction_data[var2].values
            z_values = all_cff_predictions_array[:, :, i].mean(axis=0)
            plot_3d(x_values, y_values, z_values, f'3D Plot of {cff_name} with {var1} and {var2}', f'{cff_name}_{var1}_{var2}_3d.png', 'DNN3Dimages', x_label=var1, y_label=var2, z_label=cff_name)

    # New section to plot standard deviation distribution and calculate success percentage
    success_percentages = {}
    for i, cff_name in enumerate(['ReH', 'ReE', 'ReHt', 'dvcs']):
        actual_cff = prediction_data[cff_name].values
        success_percentage = plot_std_distribution_and_calculate_success(
            actual_cff,
            all_cff_predictions_array[:, :, i],
            f'Std Dev Distribution for {cff_name}',
            f'{cff_name}_std_dev_distribution.png',
            'DNN2Dimages'
        )
        success_percentages[cff_name] = success_percentage
        print(f"Success Percentage for {cff_name}: {success_percentage:.2f}%")

    # Optionally, you can also do this for F
    success_percentage_F = plot_std_distribution_and_calculate_success(
        actual_F,
        all_f_predictions_array[:, :, 0],
        'Std Dev Distribution for F',
        'F_std_dev_distribution.png',
        'DNN2Dimages'
    )
    success_percentages['F'] = success_percentage_F
    print(f"Success Percentage for F: {success_percentage_F:.2f}%")

if __name__ == "__main__":
    main()
