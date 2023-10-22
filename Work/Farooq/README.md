# Description of the files

The directory Farooq_Code contains the code used to train, test, and validate the given data for CFFs. 

The training.sh is used to run the Python script. It will create the training and validation loss plot and save it in .png format. You can use this script on your local machine after changing the directory where you placed the code you want to run. In this script, I used a conda environment to run the code. If you do not have conda on your local machine, you can run it simply with Python after commenting the lines about the conda activation. 

The push.sh is used to push the updates on your GitHub in real time. I am generating the SSH key every time I use this script2.sh but not writing this SSH key. You can comment on these lines and use the remaining script. 

To run the shell script file execute the line: chmod +x push.sh. After this, execute the line ./push.sh. 

----------
In addition, the directory local_fit contains the local_fit code that I edited a bit to make a plot for training and validation plot and the plot is saved in the .png (sample_BKM.png). 
To run the code, you can use either localfit_updated.py or localfit_updated.sh file using bash script. To push the updates, you can use push.sh. 
.................................