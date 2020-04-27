# python3

import util

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal, glorot_normal, VarianceScaling

# Discriminator model
def discriminator_model(in_shape=(32, 32, 3)):
    init = glorot_normal()
    model = Sequential()
    model.add(Conv2D(64, (3,3), padding='same', kernel_initializer=init, input_shape=in_shape))
#     model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3,3), strides=(2,2), kernel_initializer=init, padding='same'))
#     model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3,3), strides=(2,2), kernel_initializer=init, padding='same'))
#     model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(256, (3,3), strides=(2,2), kernel_initializer=init, padding='same'))
#     model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=2e-4, beta_1=0.5),
                  metrics=['accuracy'])
    return model
