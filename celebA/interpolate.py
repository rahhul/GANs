# python3

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# generate latent points
def generate_latent_points(latent_dim, n_samples):
    # generate random points
    z = np.random.randn(latent_dim * n_samples)
    # reshape
    z = z.reshape(n_samples, latent_dim)
    return z

# uniform interpolation
# linear interpolation
