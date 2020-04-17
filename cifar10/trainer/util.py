# python3

# imports
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Versions
print("Tf version: {}".format(tf.__version__))
print("Keras version: ", keras.__version__)

# load cifar10 dataset
from tensorflow.keras.datasets.cifar10 import load_data


# load and preprocess cifar10 data
def load_cifar10():
    # load data
    (X_train, _), (_, _) = load_data()
    # convert uint8 to float32
    X = X_train.astype('float32')
    # normalise to [-1, 1]
    X = (X - 127.5) / 127.5
    return X


# label clipping/smoothing
def smooth_labels(y):
    return y - 0.3 + (np.random.random(y.shape) * 0.5)

# generate real samples
def generate_real_samples(dataset, n_samples):
    # choose randomly
    idx = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[idx]
    # create labels = 1
    y = np.ones((n_samples, 1))
    y = smooth_labels(y)
    return X, y
