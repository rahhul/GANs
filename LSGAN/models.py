# python3

import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal, glorot_normal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Flatten, Dense, Reshape, Activation
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.optimizers import Adam

in_shape = (28, 28, 1)

init = tf.keras.initializers.GlorotNormal()

# discriminator model
def build_discriminator(in_shape):
    # initialize weights
    model = Sequential()
    # downsample input to 14x14
    model.add(Conv2D(64, (4,4), strides=(2,2), padding='same',
                     kernel_initializer=init, input_shape=(in_shape)))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 7x7
    model.add(Conv2D(128, (4,4), strides=(2,2), padding='same',
                     kernel_initializer=init))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # head
    model.add(Flatten())
    model.add(Dense(1, activation='linear', kernel_initializer=init))
    # compile model
    model.compile(loss='mse', optimizer=Adam(lr=2e-4, beta_1=0.5))
    return model


# generator model
def build_generator(latent_dim):
    # initialize weights
    init = RandomNormal(stddev=0.02)
    # init = glorot_uniform()
    noise = 256 * 7 * 7
    model = Sequential()
    model.add(Dense(noise, kernel_initializer=init, input_dim=latent_dim))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    # reshape
    model.add(Reshape((7, 7, 256)))
    # upsample
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same',
                              kernel_initializer=init))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    # upsample to 28x28
    model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same',
                              kernel_initializer=init))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    # output
    model.add(Conv2D(1, (7,7), padding='same', kernel_initializer=init))
    model.add(Activation('tanh'))
    return model


# gan model
def build_gan(generator, discriminator):
    # freeze discriminator
    discriminator.trainable = False
    # stack models
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    # compile with L2 loss
    model.compile(loss='mse', optimizer=Adam(lr=2e-4, beta_1=0.5))
    return model
