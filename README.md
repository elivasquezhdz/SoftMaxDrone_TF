# SoftMaxDrone_TF

Tensorflow bynary image classification.

Dependencies:

Tensorflow, Numpy

Usage:

python softmax_dron.py

It reads the file: vectors_EDGE.npy
Wich is a numpy file containing a 485 x 900 matrix, this is the training matrix for the image classification algorythm.
It was generated with 485 images of 30 x 30 pixels each. The images were resized if needed to fit the 30x30 constraint, then edges were computed and finally the arrays flattened and converted to floats.
Once the file has been read it reads the numppy vectors in the folder: featsEDG, wich has a vector for each image to test, each vector has a shape of (1,900), because it has been pre processed as same as the training matrix (resized to 30x30, flattened and converted to float.
