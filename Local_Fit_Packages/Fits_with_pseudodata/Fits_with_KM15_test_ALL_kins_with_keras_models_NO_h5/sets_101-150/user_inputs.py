# ALWAYS CHANGE PATHS
scratch_path = '/scratch/qzf7nj/ishara_testing_only_DNN_change/sets_101-150/'

#Data file
initial_data_file = 'km15_bkm10_dropped.csv'

#Number of sets
kinematic_sets = list(range(101, 151))

#Model parameters:

Learning_Rate = 0.1
EPOCHS = 300
BATCH = 20
EarlyStop_patience = 100
modify_LR_patience = 400
modify_LR_factor = 0.9






