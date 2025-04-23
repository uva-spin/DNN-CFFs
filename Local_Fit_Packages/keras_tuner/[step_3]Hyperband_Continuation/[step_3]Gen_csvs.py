import numpy as np
import pandas as pd
import tensorflow as tf
from BHDVCS_tf_modified import *
from user_inputs import *
import matplotlib.pyplot as plt
import os
import sys

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def load_FLayer_and_cffLayer(model_path):
    LayerF = tf.keras.models.load_model(model_path, custom_objects={'TotalFLayer': TotalFLayer})
    LayerCFFs = tf.keras.Model(inputs=LayerF.input, outputs=LayerF.get_layer('cff_output_layer').output)
    return LayerF, LayerCFFs

def predict_cffs_and_f(LayerCFFs, LayerF, inputs):
    cffs = LayerCFFs.predict(inputs)
    f_values = LayerF.predict(inputs)
    return cffs, f_values

def remove_extra_columns(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    cols_to_remove = ['Mean F Prediction', 'Std Dev Prediction', 'ReH_std', 'ReE_std', 'ReHt_std', 'dvcs_std']
    rename_map = {
        "ReH_pred": "ReH", "ReE_pred": "ReE",
        "ReHt_pred": "ReHt", "dvcs_pred": "dvcs"
    }
    df.drop(columns=[c for c in cols_to_remove if c in df.columns], inplace=True)
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    df.to_csv(output_csv, index=False)

# Preprocess
remove_extra_columns(initial_data_file, "temp_pseudodata.csv")
df_all = pd.read_csv("temp_pseudodata.csv").rename(columns={"sigmaF": "errF"})

# Determine available kinematic sets
available_kin_sets = [int(d.split('_')[-1]) for d in os.listdir(scratch_path) if d.startswith('DNNmodels_Kin_Set_')]
available_kin_sets = sorted(available_kin_sets)

print(f"🧠 Evaluating on sets: {available_kin_sets}\n")

for model_index in range(10):
    print(f"\n🔎 Evaluating Top Model {model_index}...\n")

    result_df_all = pd.DataFrame()
    chi_square_dir = f'Comparison_Plots/TopModel_{model_index}'
    cff_eval_dir = f'{scratch_path}/CFFs_Evaluations/TopModel_{model_index}'
    create_folders(chi_square_dir)
    create_folders(cff_eval_dir)

    for j in available_kin_sets:
        print(f"  • Set {j}")
        kin_df = df_all[df_all['set'] == j].reset_index(drop=True)
        if kin_df.empty: continue

        models_dir = f"{scratch_path}/DNNmodels_Kin_Set_{j}"
        replica_paths = [
            os.path.join(models_dir, f) for f in os.listdir(models_dir)
            if f.endswith(f'topmodel{model_index}.h5')
        ]

        if not replica_paths:
            print(f"    ⚠️ No models found.")
            continue

        input_arr = kin_df[['QQ', 'x_b', 't', 'phi_x', 'k']].to_numpy()
        true_F = kin_df['F'].values
        phi_x = kin_df['phi_x'].values
        true_cffs = kin_df[['ReH', 'ReE', 'ReHt', 'dvcs']].iloc[0].values

        f_preds, cff_preds = [], []

        for m in replica_paths:
            LayerF, LayerCFF = load_FLayer_and_cffLayer(m)
            cffs, f_vals = predict_cffs_and_f(LayerCFF, LayerF, input_arr)
            f_preds.append(f_vals)
            cff_preds.append(cffs)

        mean_f = np.mean(f_preds, axis=0)
        std_f = np.std(f_preds, axis=0)
        chi2 = np.sum(((true_F - mean_f.ravel()) / std_f.ravel())**2)

        with open(f'{chi_square_dir}/chi_square_errors.txt', 'a') as f:
            f.write(f"{j}\t{chi2:.4f}\n")

        pd.DataFrame({
            'phi_x': phi_x,
            'Real F': true_F,
            'Mean F Prediction': mean_f.ravel(),
            'Std Dev Prediction': std_f.ravel()
        }).to_csv(f"{chi_square_dir}/F_vs_phi_x_Kinematic_Set_{j}.csv", index=False)

        # CFF histogram and summary
        plt.figure(figsize=(15, 10))
        cff_labels = ['ReH', 'ReE', 'ReHt', 'dvcs']
        cff_preds_arr = np.array(cff_preds)

        cff_summary = {
            'set': [j],
            **{f'{label}_true': [true_cffs[i]] for i, label in enumerate(cff_labels)},
            **{f'{label}_pred': [np.mean(cff_preds_arr[:, :, i])] for i, label in enumerate(cff_labels)},
            **{f'{label}_res': [abs(true_cffs[i] - np.mean(cff_preds_arr[:, :, i]))] for i, label in enumerate(cff_labels)},
            **{f'{label}_std': [np.std(cff_preds_arr[:, :, i])] for i, label in enumerate(cff_labels)}
        }

        for i, label in enumerate(cff_labels):
            plt.subplot(2, 2, i + 1)
            flat_vals = cff_preds_arr[:, :, i].flatten()
            plt.hist(flat_vals, bins=20, alpha=0.7, edgecolor='black')
            plt.axvline(x=true_cffs[i], color='red', linestyle='dashed', label='True')
            plt.title(f"{label} Predictions (Set {j})")
            plt.legend()

        plt.tight_layout()
        plt.savefig(f"{chi_square_dir}/CFF_Histograms_Set_{j}.png")
        plt.close()

        pd.DataFrame(cff_summary).to_csv(f"{cff_eval_dir}/Eval_set_{j}.csv", index=False)
        result_df_all = pd.concat([result_df_all, pd.DataFrame(cff_summary)], ignore_index=True)

    # Save global summary for this model
    result_df_all.to_csv(f"Summary_of_CFFs_TopModel_{model_index}.csv", index=False)

print("\n✅ Evaluation complete for all models.")
