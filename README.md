# Street View House Numbers (SVHN) Digit Detection and Recognition

This repository contains a deep learning model implemented in TensorFlow/Keras for the task of digit detection and recognition using the Street View House Numbers (SVHN) dataset.

The SVHN dataset contains over 600,000 digit images in the format of 32x32 RGB images. The task is to identify the digit(s) present in each image. The dataset is split into a training set and a test set, which can be downloaded from the following [link](http://ufldl.stanford.edu/housenumbers/).

## Model Architecture
The model used in this repository is a convolutional neural network (CNN) with the following architecture:

- `Input` layer (32x32 RGB image)
- `Convolutional` layer with 64 filters of size 3x3
- `Max pooling` layer with pool size of 2x2
- `Batch normalization` layer
- `Convolutional` layer with 64 filters of size 3x3
- `Max pooling` layer with pool size of 2x2
- `Batch normalization` layer
- `Convolutional` layer with 64 filters of size 3x3
- `Max pooling` layer with pool size of 2x2
- `Flatten` layer
- `Dropout` layer with a rate of 0.4
- `Output` layer with 10 units (corresponding to the 10 digits)

## Training and Evaluation
The model was trained on the training set for 40 epochs with a batch size of 128. 

The model achieved an accuracy of 92.2% on the test set.

## Acknowledgments
The SVHN dataset was accredited to Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y. Ng <ins>Reading Digits in Natural Images with Unsupervised Feature Learning</ins> *NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011*. 
