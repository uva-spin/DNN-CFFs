# Model Selection Workflow

## Step-by-Step Instructions

### Step 1 — Preprocessing (Always Run This First)
Before doing anything else, you **must run Step 1**. This step handles all necessary preprocessing and setup required for both Bayesian and Hyperband tuning workflows.

### Step 2 — Choose Tuning Strategy
After completing Step 1, you can proceed with **Step 2** using one or both of the following strategies:

- **Step 2 (Bayesian)** — Runs Bayesian Optimization for hyperparameter tuning.
- **Step 2 (Hyperband)** — Runs Hyperband for hyperparameter tuning.

Each method will generate its own set of model folders:
- `bayesian_tuning/` for Bayesian models
- `hyperband_tuning/` for Hyperband models

### Step 3 — Organize Output
Once Step 2 finishes:
- **Move** the generated folders into their respective directories:
  - Move `bayesian_tuning/` into the `bayesian/` folder.
  - Move `hyperband_tuning/` into the `hyperband/` folder.
- Then, **proceed inside each folder individually** and continue following the remaining steps from Step 1 onwards.

---

## Output Files and Final Results

After completing all steps, you will have several output files:

### `Ranked_Model_Architectures.txt`
This is the **final summary file**. It contains:
- Each model’s architecture
- The number of CFFs (Compton Form Factors) that had residuals less than 1
- The models are **ranked** in descending order based on how many small residuals they had

### `RankedLowResidualCFFModels.csv`
- This file mirrors the ranked order of `Ranked_Model_Architectures.txt`
- **Does not include** the full model architecture, making it easier to view rankings and scores at a glance

### `Summary_LowResidualCFFs_TopModel_{i}.csv`
- There will be **10 of these files** (i = 0 through 9)
- Each file corresponds to one of the **top 10 models**
- Each file includes:
  - The set number
  - Which CFFs had residuals < 1
  - The actual residual values

Other usual logs and intermediate files will also be generated during the process.

---

## Summary
1. Always start with Step 1.
2. Run Bayesian, Hyperband, or both.
3. Move generated folders into `bayesian/` and `hyperband/`.
4. Continue following folder-specific instructions.
5. Refer to `Ranked_Model_Architectures.txt` for final evaluation and model comparison.
