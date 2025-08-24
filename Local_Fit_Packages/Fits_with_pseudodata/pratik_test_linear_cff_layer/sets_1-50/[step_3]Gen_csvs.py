############################################################################
#####  Written by Ishara Fernando, Ani Venkatapuram                  #######
##############  Revised Date: 10/23/2024    ################################
##### Rivanna usage: Run the following commands on your Rivanna terminal####
## source /home/lba9wf/miniconda3/etc/profile.d/conda.sh         ###########
## conda activate env                                            ###########
## pip3 install --user tensorflow-addons==0.21.0                 ###########
############################################################################
############################################################################
### This code is only performing the evaluation              ###############
### make sure to provide the correct path to the models folder #############
############################################################################

import os
import re
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm

from BHDVCS_tf_modified import *
from user_inputs import *

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")

start_time = datetime.now()
scratch_path = str(scratch_path) + '/'
create_folders('Comparison_Plots')

def load_model_and_submodels(model_path):
    """
    Load the saved model (.keras or .h5), then construct two submodels:
      - cff_model: outputs 'cff_output_layer' (shape: [..., 4])
      - f_model:   outputs 'TotalFLayer' (linear-domain F)
    Also load the per-replica scaler if present.
    """
    full = tf.keras.models.load_model(
        model_path,
        custom_objects={'TotalFLayer': TotalFLayer},
        compile=False,
        safe_mode = False
    )

    # Submodels from named layers
    cff_model = tf.keras.Model(inputs=full.input,
                               outputs=full.get_layer('cff_output_layer').output)
    f_model = tf.keras.Model(inputs=full.input,
                             outputs=full.get_layer('TotalFLayer').output)

    # Try to load the matching scaler for this replica
    # Expect filenames like model{replica}.keras and scaler_replica_{replica}.joblib
    dirname = os.path.dirname(model_path)
    base = os.path.basename(model_path)
    m = re.search(r'model(\d+)\.(?:keras|h5)$', base)
    scaler = None
    if m:
        rid = m.group(1)
        scaler_path = os.path.join(dirname, f'scaler_replica_{rid}.joblib')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)

    return full, cff_model, f_model, scaler

def build_X8_inputs(df_subset, scaler):
    """
    Build 8-D inputs expected by the trained model:
      [QQ, x_b, t, phi_x, k, QQ_s, x_b_s, t_s]
    Uses the provided scaler (transform only). If scaler is missing, falls
    back to unscaled values for the last three dims (safe but not ideal).
    """
    raw = df_subset[['QQ', 'x_b', 't', 'phi_x', 'k']].to_numpy()
    if scaler is not None:
        scaled = scaler.transform(df_subset[['QQ', 'x_b', 't']].to_numpy())
    else:
        # Fallback: no saved scaler found (older models). Avoid refitting here.
        scaled = df_subset[['QQ', 'x_b', 't']].to_numpy()
    X8 = np.column_stack([raw, scaled]).astype(np.float32)
    return X8

def predict_cffs_and_f(cff_model, f_model, X8_inputs):
    cffs = cff_model.predict(X8_inputs, verbose=0)
    f_values = f_model.predict(X8_inputs, verbose=0)
    return cffs, f_values

def remove_extra_evaluation_columns(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    columns_to_remove = ['Mean F Prediction', 'Std Dev Prediction', 'ReH_std', 'ReE_std', 'ReHt_std', 'dvcs_std']
    existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]
    if existing_columns_to_remove:
        df.drop(columns=existing_columns_to_remove, inplace=True)

    rename_mapping = {
        "ReH_pred": "ReH",
        "ReE_pred": "ReE",
        "ReHt_pred": "ReHt",
        "dvcs_pred": "dvcs"
    }
    existing_rename_mapping = {old: new for old, new in rename_mapping.items() if old in df.columns}
    if existing_rename_mapping:
        df.rename(columns=existing_rename_mapping, inplace=True)

    df.to_csv(output_csv, index=False)
    print(f"Columns removed: {existing_columns_to_remove}")
    print(f"Columns renamed: {existing_rename_mapping}")
    print(f"File saved as {output_csv}")

# Pre-clean the input CSV
remove_extra_evaluation_columns(initial_data_file, "temp_pseudodata.csv")

# Load data
data_file = 'temp_pseudodata.csv'
df = pd.read_csv(data_file)
df = df.rename(columns={"sigmaF": "errF"})

# Choose sets to evaluate
use_specific_sets = False
specific_kin_sets = [1]

if use_specific_sets:
    available_kin_sets = []
    for kin_set in specific_kin_sets:
        folder_name = f'DNNmodels_Kin_Set_{kin_set}'
        if folder_name in os.listdir(scratch_path):
            available_kin_sets.append(kin_set)
else:
    available_kin_sets = [d for d in os.listdir(scratch_path) if d.startswith('DNNmodels_Kin_Set_')]
    available_kin_sets = [int(d.split('_')[-1]) for d in available_kin_sets]

available_kin_sets = sorted(available_kin_sets)
print(f"Available kinematic sets: {available_kin_sets}")

all_results_df = pd.DataFrame()

for j in available_kin_sets:
    print(f"Processing Kinematic Set: {j}")

    models_folder = str(scratch_path) + f'DNNmodels_Kin_Set_{j}'
    # Accept both new and old saves
    models = [os.path.join(models_folder, f) for f in os.listdir(models_folder)
              if f.endswith('.keras') or f.endswith('.h5')]

    kin_df = df[df['set'] == j].reset_index(drop=True)

    # Prepare the 8-D inputs per model (use each model's own scaler)
    real_F_values = kin_df['F'].values
    phi_x_values = kin_df['phi_x'].values
    true_values = kin_df[['ReH', 'ReE', 'ReHt', 'dvcs']].iloc[0].values

    f_predictions = []
    cff_predictions = []

    for model_path in models:
        full, cff_model, f_model, scaler = load_model_and_submodels(model_path)
        X8 = build_X8_inputs(kin_df, scaler)
        cffs, f_values = predict_cffs_and_f(cff_model, f_model, X8)
        f_predictions.append(f_values)
        cff_predictions.append(cffs)

    # Aggregate across replicas
    mean_f_predictions = []
    std_f_predictions = []
    for i in range(len(phi_x_values)):
        f_values_at_phi_x = [f_pred[i] for f_pred in f_predictions]
        mean_f_predictions.append(np.mean(f_values_at_phi_x))
        std_f_predictions.append(np.std(f_values_at_phi_x))

    mean_f_predictions = np.array(mean_f_predictions)
    std_f_predictions = np.array(std_f_predictions)

    # Chi-square (guard std=0)
    std_safe = np.where(std_f_predictions == 0, 1e-12, std_f_predictions)
    chi_square_error = np.sum(((real_F_values - mean_f_predictions) / std_safe) ** 2)

    chi_square_file = 'Comparison_Plots/chi_square_errors.txt'
    if not os.path.exists(chi_square_file):
        with open(chi_square_file, 'w') as file:
            file.write("Kinematic Set\tChi-Square Error\n")
    with open(chi_square_file, 'a') as file:
        file.write(f"{j}\t{chi_square_error:.4f}\n")
    print(f"Kinematic Set {j}: Chi-Square Error = {chi_square_error:.4f}")

    output_csv_path = f'Comparison_Plots/F_vs_phi_x_Kinematic_Set_{j}.csv'
    f_vs_phi_df = pd.DataFrame({
        'phi_x': phi_x_values,
        'Real F': real_F_values,
        'Mean F Prediction': mean_f_predictions,
        'Std Dev Prediction': std_f_predictions
    })
    f_vs_phi_df.to_csv(output_csv_path, index=False)
    print(f"F vs phi_x csvs saved: {output_csv_path}")

    # CFF histograms and CSV
    plt.figure(figsize=(15, 10))
    cff_labels = ['ReH', 'ReE', 'ReHt', 'dvcs']

    cffs_true_array = []
    cffs_pred_array = []
    cffs_stds_array = []
    csv_path = os.path.join(scratch_path, f'CFFs_Predictions_Set_{j}.csv')
    all_cff_predictions_df = pd.DataFrame()

    for i, cff_label in enumerate(cff_labels):
        plt.subplot(2, 2, i + 1)
        data = np.array(cff_predictions)[:, :, i].T.flatten()
        plt.hist(data, bins=20, edgecolor='black', alpha=0.7, color='lightblue')

        print(f"\nPredictions for {cff_label}:")
        print(data)
        print("\n" + "-" * 40)

        mean_value = np.mean(data)
        std_deviation = np.std(data)

        cffs_true_array.append(true_values[i])
        cffs_pred_array.append(mean_value)
        cffs_stds_array.append(std_deviation)

        predictions_df = pd.DataFrame({
            'set': [j] * len(data),
            'cff_label': [cff_label] * len(data),
            'prediction': data,
            'true_value': [true_values[i]] * len(data),
            'mean_value': [mean_value] * len(data),
            'std_deviation': [std_deviation] * len(data)
        })
        all_cff_predictions_df = pd.concat([all_cff_predictions_df, predictions_df], ignore_index=True)
        all_cff_predictions_df.to_csv(csv_path, index=False)
        print(f"Predictions CSV saved: {csv_path}")

    def EvalResults(j, cffs_true, cffs_pred, cffs_std):
        return pd.DataFrame({
            'set': [j],
            'ReH_true': [cffs_true[0]], 'ReH_pred': [cffs_pred[0]],
            'ReH_res': [np.abs(cffs_true[0] - cffs_pred[0])], 'ReH_std': [cffs_std[0]],
            'ReE_true': [cffs_true[1]], 'ReE_pred': [cffs_pred[1]],
            'ReE_res': [np.abs(cffs_true[1] - cffs_pred[1])], 'ReE_std': [cffs_std[1]],
            'ReHt_true': [cffs_true[2]], 'ReHt_pred': [cffs_pred[2]],
            'ReHt_res': [np.abs(cffs_true[2] - cffs_pred[2])], 'ReHt_std': [cffs_std[2]],
            'dvcs_true': [cffs_true[3]], 'dvcs_pred': [cffs_pred[3]],
            'dvcs_res': [np.abs(cffs_true[3] - cffs_pred[3])], 'dvcs_std': [cffs_std[3]],
        })

    cffs_true_array = np.array(cffs_true_array)
    cffs_pred_array = np.array(cffs_pred_array)
    cffs_stds_array = np.array(cffs_stds_array)

    create_folders(str(scratch_path) + 'CFFs_Evaluations')
    tempresults = EvalResults(j, cffs_true_array, cffs_pred_array, cffs_stds_array)
    tempresults.to_csv(str(scratch_path) + 'CFFs_Evaluations/' + f'Eval_set_{j}.csv', index=False)

    all_results_df = pd.concat([all_results_df, tempresults], ignore_index=True)

# Save combined results
all_results_df.to_csv('Summary_of_CFFs.csv', index=False)

# Merge with kinematics and F-vs-phi CSVs for final results
df_kinematic = pd.read_csv("temp_pseudodata.csv")
df_cffs = pd.read_csv("Summary_of_CFFs.csv")
df_evaluation = pd.DataFrame()

common_sets = sorted(set(df_kinematic['set']).intersection(set(df_cffs['set'])))

for set_id in common_sets:
    df_kin_set = df_kinematic[df_kinematic['set'] == set_id].copy()
    df_cffs_set = df_cffs[df_cffs['set'] == set_id].copy()
    df_f_phi = pd.read_csv(f"Comparison_Plots/F_vs_phi_x_Kinematic_Set_{set_id}.csv")

    df_cffs_repeated = pd.concat([df_cffs_set] * len(df_kin_set), ignore_index=True)

    df_kin_set.reset_index(drop=True, inplace=True)
    df_f_phi.reset_index(drop=True, inplace=True)
    df_cffs_repeated.reset_index(drop=True, inplace=True)

    df_combined = pd.concat(
        [df_kin_set, df_f_phi[['Mean F Prediction', 'Std Dev Prediction']], df_cffs_repeated],
        axis=1
    )
    df_evaluation = pd.concat([df_evaluation, df_combined], ignore_index=True)

df_evaluation['set'] = df_evaluation['set'].astype(int)

columns_to_drop = ['ReH_true', 'ReE_true', 'ReHt_true', 'dvcs_true']
df_evaluation = df_evaluation.drop(columns=columns_to_drop)

ordered_columns = [
    'set', 'k', 'QQ', 'x_b', 't', 'phi_x', 'F', 'sigmaF',
    'Mean F Prediction', 'Std Dev Prediction',
    'ReH', 'ReH_pred', 'ReH_res', 'ReH_std',
    'ReE', 'ReE_pred', 'ReE_res', 'ReE_std',
    'ReHt', 'ReHt_pred', 'ReHt_res', 'ReHt_std',
    'dvcs', 'dvcs_pred', 'dvcs_res', 'dvcs_std'
]
df_evaluation = df_evaluation[ordered_columns]
df_evaluation.to_csv("results.csv", index=False)

end_time = datetime.now()
time_file = 'time_taken_gen_csvs.txt'
elapsed = end_time - start_time
if not os.path.exists(time_file):
    with open(time_file, 'w') as file:
        file.write("Time taken for set number and replica\n")
with open(time_file, "a") as f:
    f.write(f"Start: {start_time} | End: {end_time} | Elapsed: {elapsed}\n")
