# ALWAYS CHANGE PATHS
scratch_path = '/scratch/qzf7nj/ishara_without_dropout_9_16_2025/sets_1-20_copy_20_replicas/'

#Data file
initial_data_file = 'km15_bkm10_dropped.csv'

#Number of sets
kinematic_sets = list(range(1, 21))

#Enhanced Model parameters:

Learning_Rate = 0.01  # Reduced from 0.1
EPOCHS = 2000  # Increased from 400
BATCH = 32  # Increased from 20
EarlyStop_patience = 150  # Increased from 100
modify_LR_patience = 200  # More frequent LR reduction
modify_LR_factor = 0.8  # More aggressive LR reduction







