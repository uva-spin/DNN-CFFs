import pandas as pd
from user_inputs import top_model_hparams

# Your existing Bayesian top model hyperparameters


def write_ranked_model_architectures(ranking_csv, hparams_list, output_txt='Ranked_Model_Architectures.txt'):
    df = pd.read_csv(ranking_csv)

    with open(output_txt, 'w') as f:
        for index, row in df.iterrows():
            model_str = row['model']
            model_index = int(model_str.split('_')[-1])
            count = row['low_residual_cff_count']
            hparams = hparams_list[model_index]

            f.write(f"Rank {index + 1} - {model_str} | {count} CFFs < 1\n")
            f.write("Hyperparameters:\n")
            for key, value in hparams.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n" + "-"*50 + "\n\n")

    print(f"✅ Saved model architectures to {output_txt}")

# Example usage
write_ranked_model_architectures('Ranked_LowResidualCFF_Models.csv', top_model_hparams)
