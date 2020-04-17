# python3

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

model = load_model('generator_model_050.h5')

vector = np.asarray([[0.55 for _ in range(100)]])

X = model.predict(vector)

X = (X + 1) / 2.0

plt.imshow(X[0, :, :])
plt.show()
