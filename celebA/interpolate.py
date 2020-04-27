# python3

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def slerp(val, low, high):
    """Spherical Linear Interpolation"""

    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(high),
                                     high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        # L'Hopital's rule/LERP
        return (1.0 - val) * low + val * high
    return np.sin((1.0 - val) * omega) / so * low + \
        np.sin(val * omega) / so * high


def linear_interpolation(p1: int, p2: int, n_steps: int = 10) -> np.ndarray:

    interval = np.linspace(0, 1, n_steps)
    # calculate ratios and write to ndarray
    vectors = list()
    for value in interval:
        v = slerp(value, p1, p2)
        vectors.append(v)
    return np.asarray(vectors)


# generate random points in latent space
def latent_points_interpolate(latent_dim: int, n_samples: int) -> np.ndarray:
    """ Draw random points feom a normal distribution"""

    # TODO: insert random seed
    # np.random.seed(42)
    z = np.random.randn(latent_dim * n_samples)
    # reshape
    z = z.reshape(n_samples, latent_dim)
    # interpolate
    Z = linear_interpolation(z[0], z[1])
    return Z


# plot generated images
def plot_faces(examples, n):
    for i in range(n * n):
        plt.subplot(n, n, i + 1)
        plt.axis('off')
        plt.imshow(examples[i, :, :])
    plt.show()

# RUN EXAMPLE

# load model
model = load_model('saved_models/model_40.h5')
n = 20
results = None
# generate poitns in latent space and interpolate
for i in range(0, n, 2):
    interpolated_points = latent_points_interpolate(100, n)

    X = model.predict(interpolated_points)
    X = (X + 1) / 2.0
    if results is None:
        results = X
    else:
        results = np.vstack((results, X))

plot_faces(results, 10)
