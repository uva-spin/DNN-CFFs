import pandas as pd
import numpy as np

# === Input files ===
dnn_exp_csv = "DNN_projections.csv"      # Blue: DNN fit to experimental data
dnn_km15_csv = "results.csv"             # Green: DNN fit to KM15 pseudodata
output_csv = "CFF_consistency_report.csv"

# === Load data ===
df_exp = pd.read_csv(dnn_exp_csv)
df_km15 = pd.read_csv(dnn_km15_csv)

# === Define CFFs of interest ===
cff_labels = ["ReH", "ReHt", "ReE", "dvcs"]

# === Prepare sets ===
sets = sorted(list(set(df_exp["set"]).intersection(df_km15["set"])))
print(f"Found {len(sets)} overlapping kinematic sets.")

# === Track results ===
results = []

for set_num in sets:
    row_exp = df_exp[df_exp["set"] == set_num].iloc[0]
    row_km15 = df_km15[df_km15["set"] == set_num].iloc[0]

    all_within = True
    cff_results = {}

    for cff in cff_labels:
        true_val = row_km15[cff]                # Red: KM15 truth
        blue_val = row_exp[f"{cff}_pred"]       # Blue: DNN (exp)
        green_pred = row_km15[f"{cff}_pred"]    # Green: DNN (KM15)
        green_std = row_km15[f"{cff}_std"]

        lower = green_pred - green_std
        upper = green_pred + green_std

        km15_inside = lower <= true_val <= upper
        dnn_inside = lower <= blue_val <= upper
        both_inside = km15_inside and dnn_inside

        cff_results[cff] = both_inside
        if not both_inside:
            all_within = False

    results.append({
        "set": set_num,
        **{f"{cff}_inside": cff_results[cff] for cff in cff_labels},
        "all_CFFs_consistent": all_within
    })

# === Save results ===
df_results = pd.DataFrame(results)
df_results.to_csv(output_csv, index=False)

# === Summary ===
consistent_sets = df_results[df_results["all_CFFs_consistent"] == True]["set"].tolist()
inconsistent_sets = df_results[df_results["all_CFFs_consistent"] == False]["set"].tolist()

print("\n=== CFF Consistency Check ===")
print(f"Consistent sets (all 4 CFFs within green's std): {len(consistent_sets)}")
print(f"Inconsistent sets: {len(inconsistent_sets)}")

if consistent_sets:
    print("\nConsistent set numbers:")
    print(consistent_sets)

print(f"\nDetailed report saved to {output_csv}")
