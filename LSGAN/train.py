# python3

from utils import generate_mnist_samples, generate_fake_samples
from utils import generate_latent_points, load_mnist
from models import build_generator, build_discriminator, build_gan

import matplotlib.pyplot as plt
import numpy as np

# summarize performance
def log_performance(step, g_model, latent_dim, n_samples=100):
    # generate fake samples for evaluation
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # preprocess pixel values ?
    X = (X + 1) / 2.0
    # plot images
    for i in range(10 * 10):
        # subplots
        plt.subplot(10, 10, i+1)
        plt.axis('off')
        plt.imshow(X[i, :, :, 0])
    # save plot to disk
    filename = "plot_at_%06d.png" % (step + 1)
    plt.savefig(filename)
    plt.close()
    # save model
    model_name = "model_%06d.h5" % (step + 1)
    g_model.save(model_name)
    print(f"Saved {filename} and {model_name}")


# plot performance of model
def plot_history(d1_hist, d2_hist, g_hist):
    plt.plot(d1_hist, label='dloss1')
    plt.plot(d2_hist, label='dloss2')
    plt.plot(g_hist, label='gloss')
    plt.legend()
    filename = "plot_loss.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


# main training loop
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=10,
          n_batch=64):
    # calculate number of batches per epoch
    batch_per_epoch = int(dataset.shape[0] / n_batch)
    # no of training iterations
    n_steps = batch_per_epoch * n_epochs
    # half-batch
    half_batch = n_batch // 2
    # empty lists to store loss
    d1_hist, d2_hist, g_hist = list(), list(), list()
    # training loop
    for i in range(n_steps):
        # generate real and fake samples
        X_real, y_real = generate_mnist_samples(dataset, half_batch)
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        # update generator
        z_input = generate_latent_points(latent_dim, n_batch)
        y_real2 = np.ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(z_input, y_real2)
        #
        print(">%d, d1=%.3f, d2=%.3f, g=%.3f" % (i+1, d_loss1, d_loss2, g_loss))
        # record
        d1_hist.append(d_loss1)
        d2_hist.append(d_loss2)
        g_hist.append(g_loss)
        # evaluate
        if (i+1) % (batch_per_epoch * 1) == 0:
            log_performance(i, g_model, latent_dim)
    # plot
    plot_history(d1_hist, d2_hist, g_hist)



# EXAMPLE

latent_dim = 100

# discriminator model
discriminator = build_discriminator(in_shape=(28, 28, 1))

# generator model
generator = build_generator(latent_dim=latent_dim)

# gan model
gan_model = build_gan(generator, discriminator)

# image dataset
dataset = load_mnist()
print(dataset.shape)

# train

train(generator, discriminator, gan_model, dataset, latent_dim)
