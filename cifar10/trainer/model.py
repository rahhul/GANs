# python3

from os import makedirs

# from comet_ml import Experiment
import util, generator, discriminator

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam


# # COMET ML
# my_api_key = os.environ.get('comet_api_key', None)

# # Add the following code anywhere in your machine learning file
# experiment = Experiment(api_key=my_api_key, project_name="exp", workspace="rahhul")


# Define GAN
def define_gan(g_model, d_model):
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    # compile
    opt = Adam(lr=2e-4, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


# save plot of generated images
def save_plot(examples, epoch, n=10):
    # scale from [-1, 1] to [0, 1]
    examples = (examples + 1) / 2.0
    # plot
    for i in range(n*n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i])
    # save plot to file
    filename = 'results_glorot_normal/plot_g%03d.png' % (epoch + 1)
    plt.savefig(filename)
    plt.close()


# performance to console
def eval_performance(epoch, g_model, d_model, dataset,
                          latent_dim, n_samples=150):
    # real samples
    X_real, y_real = util.generate_real_samples(dataset, n_samples)
    # evaluate discriminator
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # fake samples
    X_fake, y_fake = generator.generate_fake_samples(g_model, latent_dim,
                                                     n_samples)
    # evaluate
    _, acc_fake = d_model.evaluate(X_fake, y_fake, verbose=0)
    # print to console
    print(">>Accuracy Real: %.0f%%, Fake: %.0f%%" % (acc_real*100, acc_fake*100))
    # save plot
    save_plot(X_fake, epoch)
    # save model
    filename = "results_glorot_normal/model_%03d.h5" % (epoch + 1)
    g_model.save(filename)

# train
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=60,
          n_batch=256):
    batch_per_epoch = int(dataset.shape[0] / n_batch) # 195
    half_batch = int(n_batch / 2)
    # enumerate over epochs
    for i in range(n_epochs):
        # enumerate over training set
        for j in range(batch_per_epoch):
            # get real samples
            X_real, y_real = util.generate_real_samples(dataset, half_batch)
            # update discriminator
            d_loss1, d_acc1 = d_model.train_on_batch(X_real, y_real)
            # generate fake samples
            X_fake, y_fake = generator.generate_fake_samples(g_model,
                                                             latent_dim,
                                                             half_batch)
            # update discriminator
            d_loss2, d_acc2 = d_model.train_on_batch(X_fake, y_fake)
            # points in latent space
            X_gan = generator.generate_latent_points(latent_dim, n_batch)
            # inverted labels
            y_gan = np.ones((n_batch, 1))
            # update gan
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # print performance to console
            print(">>Epoch:%d, %d/%d, d_loss_real=%.2f, d_loss_fake=%.3f, gan_loss=%.3f, acc-real=%d, acc-fake=%d" %
                  (i+1, j+1, batch_per_epoch, d_loss1, d_loss2, g_loss, int(d_acc1*100), int(d_acc2*100)))
        if (i + 1) % 5 == 0:
            eval_performance(i, g_model, d_model, dataset, latent_dim)


# RUN TRAINING

makedirs('results_glorot_normal', exist_ok=True)

latent_dim = 128

# create a discriminator
d_model = discriminator.discriminator_model()

# create a generator
g_model = generator.generator_model(latent_dim)

# gan
gan_model = define_gan(g_model, d_model)

dataset = util.load_cifar10()

train(g_model, d_model, gan_model, dataset, latent_dim)
