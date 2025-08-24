# ALWAYS CHANGE PATHS
scratch_path = '/scratch/qzf7nj/tanh_pratik_smaller_151-195/'

#Data file
initial_data_file = 'km15_bkm10_dropped.csv'

#Number of sets
kinematic_sets = list(range(151, 196))

#Model parameters:

# Model parameters (match "fast" script behavior)
Learning_Rate = 0.1     # Adam(1e-1) in the fast code
EPOCHS = 2000            # fast code trained up to 300 epochs with early stopping
BATCH = 1               # batch_size=1 in the fast code
EarlyStop_patience = 50
modify_LR_patience = 40
modify_LR_factor = 0.8






