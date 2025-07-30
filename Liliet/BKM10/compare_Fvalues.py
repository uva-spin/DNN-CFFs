import pandas as pd
import tensorflow as tf
from BKM_DVCS_8Pars import BKM_DVCS  # Adjust if your class is in a different module

# Load input CSV file
input_file = "/Users/liliet/Library/CloudStorage/OneDrive-LosAlamosNationalLaboratory/R-KMI/Liliet/Pseudodata/pseudo_KM15_BKM10_Jlab_all_t2_8pars.csv"
df = pd.read_csv(input_file)

# Initialize the BKM_DVCS model
bkm = BKM_DVCS()

# Prepare input tensors
kins = tf.constant(df[["k", "QQ", "xB", "t", "phi"]].values, dtype=tf.float32)
cffs = tf.constant(df[["ReH", "ReE", "ReHt", "ReEt", "ImH", "ImE", "ImHt", "ImEt"]].values, dtype=tf.float32)

# Compute XS
xs = bkm.total_xs(kins, cffs, twist="t2")
xs = xs.numpy().flatten()  # Convert to NumPy array for saving

# Add predictions to DataFrame
df["F_true_py"] = xs

# Prepare input for total_xs_fix_dvcs
pars_fix = tf.constant(df[["ReH", "ReE", "ReHt", "dvcs"]].values, dtype=tf.float32)

# Compute total_xs_fix_dvcs that uses the dvcs value from the pseudodata file
xs_fix = bkm.total_xs_fix_dvcs(kins, pars_fix, twist="t2")
df["F_true_fix_dvcs"] = xs_fix.numpy().flatten()

# Define output columns
output_columns = [
    "k", "QQ", "xB", "t", "phi",
    "ReH", "ReE", "ReHt", "ReEt", "ImH", "ImE", "ImHt", "ImEt",
    "F_true", "F_true_py", "F_true_fix_dvcs"
]
output_df = df[output_columns]

# Save to CSV
output_file = "F_true_comparison_t2.csv"
output_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")