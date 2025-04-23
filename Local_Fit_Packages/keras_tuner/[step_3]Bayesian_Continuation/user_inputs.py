# ALWAYS CHANGE PATHS
scratch_path = '/scratch/<user_id>/bayesian_models_test/'

#Data file
initial_data_file = '../Pseudo_data_with_sampling.csv'

#Number of sets
kinematic_sets = list(range(1, 196))




#Model parameters:

Learning_Rate = 0.01
EPOCHS = 2000
BATCH = 20
EarlyStop_patience = 1000
modify_LR_patience = 400
modify_LR_factor = 0.9






