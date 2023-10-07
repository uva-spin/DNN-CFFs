# Changes made by Vyas, Chinmay, Nuwan, and Ani - 10/6/23

## localfit_updated.py

* for loop eliminated so that all of the data is fed into the model at once - Vyas and Nuwan
* kinematic *k* is not fed into the training inputs of the model - Chinmay and Ani

## generate_pseudodata.py - Vyas and Nuwan

* script written to read kinematics and cffs from csv
* *F1* and *F2* are calculated based on input data
* *F* is calculated based on input data
* *phi* is varied to produce different values of *F* depending on angle measure
* resulting pseudodata is written to csv

## utils.py

* class structure allows for multiple models/methods to be implemented

### F_calc - Vyas and Nuwan

* function from *BHDVCS_tf.py* is copy-pasted as a function to calculate *F*

### F1F2 - Chinmay and Ani

* c++ code to calculate *F1* and *F2* is translated to python

## Models - Chinmay and Ani

* model from localfit_v2.py is added here as one model
* input shape is 3 (to exclude *k*)