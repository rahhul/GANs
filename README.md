---
description: Notes and experiments on GANs.
---

# GANs

Tensorflow implementation of DCGANs \[Generative Adversarial Networks\]. Models trained on GCP with 1 x NVIDIA Tesla T4.

![Cifar 10 DCGAN images after 60 epochs](.gitbook/assets/plot_g060.png)

CIFAR-10 and CelebA DCGAN generated images.

![CelebA DCGAN images after 60 epochs](.gitbook/assets/plot_e060.png)



## DCGAN on CelebA dataset

#### **\# TODO: Merge with Master**

### Experiments:

* [x] Label Clipping/Label Smoothing
* [x] BatchNorm layers in Generator model
* [ ] Change plotting and integrate Comet ML
* [ ] Streamlit.io 

–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

### To Do:

* [x] MTCNN implementation for "face extraction"
* [x] Create np.savez\_compressed
* [ ] Tf-Gan model abstraction
* [ ] Latent Space _SLERP_ interpolation



## DCGAN on Cifar10

#### kernel initializer = glorot\_normal



### 









