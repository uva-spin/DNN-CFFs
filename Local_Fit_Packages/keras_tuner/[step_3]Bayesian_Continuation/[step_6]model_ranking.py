import pandas as pd
import os

def rank_models_by_low_residuals(top_n_models=10, output_file='Ranked_LowResidualCFF_Models.csv'):
    results = []

    for i in range(top_n_models):
        file_path = f'Summary_LowResidualCFFs_TopModel_{i}.csv'
        if not os.path.exists(file_path):
            print(f"⚠️ Skipping missing file: {file_path}")
            continue

        df = pd.read_csv(file_path)
        total_cffs = 0

        for _, row in df.iterrows():
            # Count how many CFFs are in the comma-separated string
            cffs = row['low_residual_cffs']
            count = len(cffs.split(',')) if isinstance(cffs, str) else 0
            total_cffs += count

        results.append({'model': f'top_model_{i}', 'low_residual_cff_count': total_cffs})

    # Create final ranking
    ranking_df = pd.DataFrame(results)
    ranking_df.sort_values(by='low_residual_cff_count', ascending=False, inplace=True)
    ranking_df.reset_index(drop=True, inplace=True)

    ranking_df.to_csv(output_file, index=False)
    print(f"✅ Ranked summary saved to {output_file}")

# Run it
rank_models_by_low_residuals()
