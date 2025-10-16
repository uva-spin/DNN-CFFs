# ALWAYS CHANGE PATHS
scratch_path = '/scratch/qzf7nj/10_2_trial_models_global_fit/layer_relu6_tanh/'

#Data file
initial_data_file = 'km15_bkm10_dropped.csv'

#Number of sets
kinematic_sets = list(range(21, 65))

#Enhanced Model parameters:

Learning_Rate = 0.01  # Reduced from 0.1
EPOCHS = 2000  # Increased from 400
BATCH = 64  # Increased from 20
EarlyStop_patience = 150  # Increased from 100
modify_LR_patience = 200  # More frequent LR reduction
modify_LR_factor = 0.8  # More aggressive LR reduction







