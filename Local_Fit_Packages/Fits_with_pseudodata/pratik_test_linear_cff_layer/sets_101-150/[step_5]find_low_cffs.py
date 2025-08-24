import pandas as pd
from collections import Counter
################################################################
# 2 files produced: Summary_LowResidualCFFs_from_results.csv
# CFF_LowResidual_Counts.csv
################################################################

def summarize_low_residual_cffs_from_results(threshold=2.5):
    """
    Reads results.csv, identifies CFFs in each kinematic set
    with residuals below the threshold, and outputs a summary CSV.

    Output format:
    set,low_residual_cffs,residuals
    101,"ReH, ReHt","0.28, 0.72"
    102,"ReE","0.87"
    """
    input_file = 'results.csv'
    output_file = 'Summary_LowResidualCFFs_from_results.csv'
    cff_labels = ['ReH', 'ReE', 'ReHt', 'dvcs']

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"File not found: {input_file}")
        return

    summary_rows = []

    for set_id in sorted(df['set'].unique()):
        row = df[df['set'] == set_id].iloc[0]  
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
        result_df.to_csv(output_file, index=False)
        print(f" Saved: {output_file} with {len(result_df)} rows")
    else:
        print(f"No residuals below {threshold} found in {input_file}")

summarize_low_residual_cffs_from_results()


def count_low_residual_cffs(summary_file='Summary_LowResidualCFFs_from_results.csv'):
    """
    Reads the summary CSV of low residual CFFs and counts total appearances of each CFF.
    
    Output format:
    CFF,count
    ReH,23
    ReE,19
    ...
    """
    try:
        df = pd.read_csv(summary_file)
    except FileNotFoundError:
        print(f"File not found: {summary_file}")
        return

    cff_counter = Counter()

    for _, row in df.iterrows():
        if pd.notna(row['low_residual_cffs']):
            cffs = [cff.strip() for cff in row['low_residual_cffs'].split(',')]
            cff_counter.update(cffs)

    count_df = pd.DataFrame(cff_counter.items(), columns=['CFF', 'count'])
    count_df = count_df.sort_values(by='count', ascending=False)

    total = count_df['count'].sum()
    total_row = pd.DataFrame([{'CFF': 'Total', 'count': total}])
    count_df = pd.concat([count_df, total_row], ignore_index=True)

    output_file = 'CFF_LowResidual_Counts.csv'
    count_df.to_csv(output_file, index=False)
    print(f" Saved CFF count summary with total to {output_file}")

count_low_residual_cffs()

