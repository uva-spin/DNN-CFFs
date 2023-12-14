#!/bin/bash

validate_integer() {
if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo -e "\033[0;31m$1\033[0m"
    return 1
  else
    echo -e "\033[0;32m$1\033[0m"
    return 0
  fi
}

validate_float() {
if ! [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo -e "\033[0;31m$1\033[0m"
    return 1
  else
    echo -e "\033[0;32m$1\033[0m"
    return 0
  fi
}

validate_string() {
if ! [[ "$1" =~ ^[a-zA-Z]+$ ]]; then
    echo -e "\033[0;31$1\033[0m"
    return 1
  else
    echo -e "\033[0;32m$1\033[0m"
    return 0
  fi
}

confirm_values() {
  echo "Below are the values you are using for training your model:"
  echo -n "Number of Jobs: "
  validate_integer $NJOBS
  echo -n "Number of Hidden Layers: "
  validate_integer $num_hidden_layers
  echo -n "Number of Nodes: "
  validate_integer $num_nodes
  echo -n "Activation Function: "
  validate_string $activation_function
  echo -n "Learning Rate: "
  validate_float $learning_rate
  echo -n "Patience for Early Stop: "
  validate_integer $EarlyStop_patience
  echo -n "Patience for Modify LR: "
  validate_integer $modify_LR_patience
  echo -n "Factor for Modify LR: "
  validate_float $modify_LR_factor
}

read_input() {
  read -p "Enter the Number of Jobs " NJOBS
  read -p "Enter the Number of nodes or leave blank for default value  " input_num_hidden_layers
  num_hidden_layers=${input_num_hidden_layers:-2}
  read -p "Enter the Number of nodes or leave blank for default value " input_num_nodes
  num_nodes=${input_num_nodes:-300}
  read -p "Enter the Number of nodes or leave blank for default value  " input_activation_function
  activation_function=${input_activation_function:-relu}
  read -p "Enter the Number of nodes or leave blank for default value  " input_learning_rate
  learning_rate=${input_learning_rate:-0.0001}
  read -p "Enter the Number of nodes or leave blank for default value  " input_earlystop_patience
  EarlyStop_patience=${input_earlystop_patience:-1000}
  read -p "Enter the Number of nodes or leave blank for default value  " input_modify_lr_patience
  modify_LR_patience=${input_modify_lr_patience:-400}
  read -p "Enter the Number of nodes or leave blank for default value " input_modify_lr_factor
  modify_LR_factor=${input_modify_lr_factor:-0.9}
}

read_input
confirm_values
while true; do
  read -p "Do these values look good? (yes/no): " answer
  case $answer in
    [Yy]* ) break;;
    [Nn]* ) read_input; confirm_values;;
    * ) echo "Please answer yes or no.";;
  esac
done

SCRIPT_NAME="LMIFIT.py"

# Submit Jobs
for (( i=1; i<=$NJOBS; i++)); do
  sbatch grid.slurm $SCRIPT_NAME $i $num_nodes $learning_rate $activation_function $EarlyStop_patience $modify_LR_patience $modify_LR_factor $num_hidden_layers 
done
