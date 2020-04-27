# python3

import numpy as np
from models import generator_model, discriminator_model, gan_model
import matplotlib.pyplot as plt
# from tensorflow.keras.utils import Progbar
from os import makedirs


# load and preprocess training images
def load_real_samples():
    data = np.load('img_align_celeba.npz')
    X = data['arr_0']
    # convert from uint to float32
    X = X.astype('float64')
    X = (X - 127.5) / 127.5
    return X


# label smoothing
def smooth_labels(y):
    return y - 0.3 + (np.random.random(y.shape) * 0.5)


# generate samples
def generate_real_samples(dataset, n_samples):
    # random index
    idx = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[idx]
    # labels = 1
    y = np.ones((n_samples, 1))
    y = smooth_labels(y)
    return X, y


# inputs to the generator from the latent space
def generate_latent_points(latent_dim, n_samples):
    # generate random normal points
    x_input = np.random.randn(latent_dim * n_samples).reshape(n_samples,
                                                              latent_dim)
    return x_input


# use generator model tp generate fake samples
def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    # feed x_input into generator model
    X = g_model(x_input, training=False)
    # create fake labels = 0
    y = np.zeros((n_samples, 1))
    return X, y


# save plots
def save_plot(examples, epoch, n=10):
    # scale images from [-1, 1] to [0, 1]
    examples = (examples + 1) / 2.0
    # plot images
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis("off")
        plt.imshow(examples[i])
    # save file
    filename = 'results/plot_e%03d.png' % (epoch + 1)
    plt.savefig(filename)
    plt.close()


def evaluate_performance(epoch, g_model, d_model, dataset, latent_dim,
                         n_samples=100):
    # Real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # Fake samples
    X_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator
    _, acc_fake = d_model.evaluate(X_fake, y_fake, verbose=0)
    # print performane to console
    print("Acc Real: %.0f%%, Acc Fake: %.0f%%" % (acc_real*100, acc_fake*100))
    # generate a plot of the images
    save_plot(X_fake, epoch)
    # save model
    model_filename = 'results/model_%02d.h5' % (epoch + 1)
    g_model.save(model_filename)


# Train function

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100,
          n_batch=128):

    batch_per_epoch = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # enumerate epochs
    for i in range(n_epochs):
        for j in range(batch_per_epoch):
            # get random real samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # generate fake samples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim,
                                                   half_batch)
            # update discriminator again
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # points in latent space
            X_gan = generate_latent_points(latent_dim, n_batch)
            # inverted labels
            y_gan = np.ones((n_batch, 1))
            # update gan
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # print performance to console
            print(">>%d, %d/%d, d_loss_real=%.2f, d_loss_fake=%.2f, gan=%.2f" %
                  (i+1, j+1, batch_per_epoch, d_loss1, d_loss2, g_loss))
        # evaluate performance
        if (i+1) % 5 == 0:
            evaluate_performance(i, g_model, d_model, dataset, latent_dim)


# RUN
makedirs('results', exist_ok=True)

latent_dim = 100

d_model, g_model = discriminator_model(), generator_model(latent_dim)

gan = gan_model(g_model, d_model)

dataset = load_real_samples()

# train model
train(g_model, d_model, gan, dataset, latent_dim)
