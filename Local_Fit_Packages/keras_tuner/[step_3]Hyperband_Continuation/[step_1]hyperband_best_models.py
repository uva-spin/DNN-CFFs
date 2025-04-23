import keras_tuner as kt
import re
import os

# Set paths
directory = "hyperband_tuning"
project_name = "dnn_optimization"
user_inputs_path = "user_inputs.py"

# Load the tuner
tuner = kt.Hyperband(
    hypermodel=None,
    objective="val_loss",
    max_epochs=3000,
    factor=4,
    directory=directory,
    project_name=project_name
)
tuner.reload()

# Extract top 10 hyperparameters
top_hparams = tuner.get_best_hyperparameters(num_trials=10)
top_hparams_dicts = [hp.values for hp in top_hparams]

# Format for Python file
formatted_output = "top_model_hparams = [\n"
for entry in top_hparams_dicts:
    formatted_output += f"    {entry},\n"
formatted_output += "]\n"

# Read and update user_inputs.py
if os.path.exists(user_inputs_path):
    with open(user_inputs_path, "r") as file:
        content = file.read()
else:
    content = ""

if "top_model_hparams" in content:
    content = re.sub(
        r"top_model_hparams\s*=\s*\[.*?\]\n",
        formatted_output,
        content,
        flags=re.DOTALL
    )
else:
    content += "\n\n" + formatted_output

with open(user_inputs_path, "w") as file:
    file.write(content)

print("Updated user_inputs.py with top 10 Hyperband models as 'top_model_hparams'.")
