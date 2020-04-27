# python3

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, LeakyReLU, Flatten
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_normal


# define discriminator
def discriminator_model(in_shape=(80, 80, 3)):
    model = Sequential()
    init = glorot_normal()
    # Input
    model.add(Conv2D(128, (5, 5), padding='same', kernel_initializer=init,
                     input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 40x40
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same',
                     kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same',
                     kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same',
                     kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same',
                     kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    # flatten and classify
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    # compile
    opt = Adam(lr=2e-4, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    return model


# define generator
def generator_model(latent_dim):
    n_nodes = 128 * 5 * 5
    model = Sequential()
    init = glorot_normal()
    # Input
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((5, 5, 128)))
    # upsample
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same',
                              kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    # upsample
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same',
                              kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    # upsample
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same',
                              kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    # upsample
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same',
                              kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    # output layer
    model.add(Conv2D(3, (5, 5), activation='tanh', padding='same',
                     kernel_initializer=init))
    return model


# GAN model
def gan_model(g_model, d_model):
    # freeze discriminator
    d_model.trainable = False
    # model
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    # compile
    opt = Adam(lr=2e-4, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model
