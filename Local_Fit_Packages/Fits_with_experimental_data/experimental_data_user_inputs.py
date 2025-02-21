
###### ALWAYS CHANGE ########
scratch_path = '/scratch/<computing-id>/DNN_CFFs/<whatever_path_of_choice>'
##### ALWAYS CHANGE ABOVE #######


#### User's inputs ####
Learning_Rate = 0.001
EPOCHS = 2000
BATCH = 20
EarlyStop_patience = 1000
modify_LR_patience = 400
modify_LR_factor = 0.9

# You can modify the following list to include the sets you want to run
# This list can be modified dynamically
kinematic_sets = list(range(139, 151))


