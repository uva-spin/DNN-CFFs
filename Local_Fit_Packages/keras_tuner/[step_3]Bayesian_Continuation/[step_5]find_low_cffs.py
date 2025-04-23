import pandas as pd
import os

def summarize_low_residual_cffs(top_n_models=10, threshold=1.0):
    """
    For each Summary_of_CFFs_TopModel_{i}.csv file, identify the CFFs in each kinematic set
    with residuals below the threshold, and output a clean CSV summary.
    
    Output format per model:
    set,low_residual_cffs,residuals
    101,"ReH, ReHt","0.28, 0.72"
    102,"ReE","0.87"
    """
    cff_labels = ['ReH', 'ReE', 'ReHt', 'dvcs']

    for model_index in range(top_n_models):
        summary_file = f'Summary_of_CFFs_TopModel_{model_index}.csv'
        if not os.path.exists(summary_file):
            print(f"⚠️ File not found: {summary_file}")
            continue

        print(f"📂 Processing: {summary_file}")
        df = pd.read_csv(summary_file)
        
        summary_rows = []

        for _, row in df.iterrows():
            set_id = int(row['set'])
            cffs_under_threshold = []
            residuals_under_threshold = []

            for cff in cff_labels:
                res_col = f"{cff}_res"
                if res_col in row and row[res_col] < threshold:
                    cffs_under_threshold.append(cff)
                    residuals_under_threshold.append(round(row[res_col], 4))

            if cffs_under_threshold:
                summary_rows.append({
                    'set': set_id,
                    'low_residual_cffs': ', '.join(cffs_under_threshold),
                    'residuals': ', '.join(str(val) for val in residuals_under_threshold)
                })

        if summary_rows:
            result_df = pd.DataFrame(summary_rows)
            output_file = f'Summary_LowResidualCFFs_TopModel_{model_index}.csv'
            result_df.to_csv(output_file, index=False)
            print(f"✅ Saved: {output_file} with {len(result_df)} rows")
        else:
            print(f"ℹ️ No residuals below {threshold} found for model {model_index}")


# Run it!
summarize_low_residual_cffs()
