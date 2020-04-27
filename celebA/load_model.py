# python3

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# generate latent points from a Gaussian distribution
def generate_latent_points(latent_dim, n_samples):
    # generate random points
    latent_points = np.random.randn(latent_dim * n_samples)
    # reshape
    latent_points = latent_points.reshape(n_samples, latent_dim)
    return latent_points

# plot figures
def plot_model_output(predictions, n):
    # plot
    for i in range(n*n):
        plt.subplot(n, n, 1 + i)
        # axis off
        plt.axis('off')
        plt.imshow(predictions[i, :, :])
    plt.show()

# RUN MODEL
model = load_model('saved_models/model_40.h5')

latent_points = generate_latent_points(100, 25)

model_output = model.predict(latent_points)

# scale pixel values to [0, 1]
model_output = (model_output + 1) / 2.0

plot_model_output(model_output, 3)
