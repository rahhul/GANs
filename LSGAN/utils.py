import tensorflow as tf
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

print(f"Tf version: {tf.__version__}")


# import and load mnist

from tensorflow.keras.datasets.mnist import load_data

# retrieve and load mnist images
def load_mnist():
    """Load and scale mnist data
    """
    (X_train, _), (_,_) = load_data()
    # add channels dimension
    X = np.expand_dims(X_train, axis=-1)
    X = X.astype('float32')
    # scale pixel values
    X = (X - 127.5) / 127.5
    return X

# generate real mnist data with label == 1
def generate_mnist_samples(dataset, n_samples):
    """Generate samples for the discriminator"""
    # choose random index
    idx = np.random.randint(0, dataset.shape[0], n_samples)
    # select images
    X = dataset[idx]
    # labels == 1
    y = np.ones((n_samples, 1))
    return X, y


# Latent Points
def generate_latent_points(latent_dim, n_samples):
    """Generate random noise as input
    to the Generator"""
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape as batches to generator
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# generate fake samples
def generate_fake_samples(generator, latent_dim, n_samples):
    """Generator will predict fake samples"""
    x_input = generate_latent_points(latent_dim, n_samples)
    # feed this noise into the generator model
    X = generator.predict(x_input)
    # concat with labels == 0
    y = np.zeros((n_samples, 1))
    return X, y
