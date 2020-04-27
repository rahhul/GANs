# python3

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2DTranspose, LeakyReLU, Reshape
from tensorflow.keras.layers import BatchNormalization, Conv2D
from tensorflow.keras.initializers import RandomNormal, glorot_normal, VarianceScaling

# Generator model
def generator_model(latent_dim):
    init = glorot_normal()
    model = Sequential()
    # nodes
    n_nodes = 256 * 4 * 4
    model.add(Dense(n_nodes, kernel_initializer=init, input_dim=latent_dim))
#     model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 256)))
    # upsample to 8x8
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), kernel_initializer=init, padding='same'))
#     model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 16x16
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), kernel_initializer=init, padding='same'))
#     model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), kernel_initializer=init, padding='same'))
#     model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # output layer
    model.add(Conv2D(3, (3,3), activation='tanh', kernel_initializer=init, padding='same'))
    return model



# generate latent points
def generate_latent_points(latent_dim, n_samples):
    # draw from a gaussian distribution
    x_input = x_input = np.random.randn(latent_dim * n_samples)
    # reshape
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    # feed into g_model
    X = g_model.predict(x_input)
    # labels
    y = np.zeros((n_samples, 1))
    return X, y
