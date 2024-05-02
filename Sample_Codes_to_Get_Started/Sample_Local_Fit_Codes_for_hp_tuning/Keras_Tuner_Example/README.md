There is a .slurm file in this folder which you can use to run jobs on Rivanna

Note: 
Run the following code to install keras_tuner with tensorflow version 2.13.0. Also, remember to check the code whether you have changed 'tensorflow' to 'tensor_flow' in the import if you are running the code on Rivanna with version 2.13.0.

 ### apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0sif -m pip install --user keras_tuner 
