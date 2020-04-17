# python3

# Load Gan trained model from directory to see results

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# generate latent points
# generate latent points
def generate_latent_points(latent_dim, n_samples):
    # draw from a gaussian distribution
    x_input = x_input = np.random.randn(latent_dim * n_samples)
    # reshape
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# create and display a plot
def display_plot(examples, n):
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis("off")
        plt.imshow(examples[i, :, :])
    plt.show()


model = load_model('generator_model_050.h5')
latent_points = generate_latent_points(100, 100)
X = model.predict(latent_points)
print(X.shape)
# scale X
X = (X + 1) / 2.0

display_plot(X, 10)
