# ALWAYS CHANGE PATHS
scratch_path = '/scratch/qzf7nj/10_replicas_10_kinematics_km15_simple_architecture/'

#Data file
initial_data_file = 'km15_bkm10_dropped.csv'

#Number of sets
kinematic_sets = list(range(1, 11))

#Model parameters:

Learning_Rate = 0.01
EPOCHS = 2000
BATCH = 20
EarlyStop_patience = 1000
modify_LR_patience = 400
modify_LR_factor = 0.9






