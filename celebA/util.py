# python3

# Load and preprocess CelebA images
from os import listdir
import pathlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import mtcnn
import cv2

detector = mtcnn.MTCNN()
# load image
def decode_img(filename):
    image = tf.io.read_file(filename)
    # convert
    image = tf.image.decode_jpeg(image)
    return image

# Function to extract faces from dataset
# See: https://github.com/edumucelli/mtcnn/blob/master/example.py
def extract_face(model, pixels, size=(80, 80)):
    # detect faces
    faces = detector.detect_faces(pixels)
    # skip if no face present
    if len(faces) == 0:
        return None
    # extract details of the face
    x1, y1, width, height = faces[0]['box']
    x1, y1 = abs(x1), abs(y1)
    # convert to coordinates
    x2, y2 = x1 + width, y1 + height
    # retrieve face pixels
    face_pixels = pixels[y1:y2, x1:x2]
    # resize
    image = Image.fromarray(face_pixels)
    image = image.resize(size)
    face_array = np.asarray(image)
    return face_array



# load and extract faces for all images
def load_faces(directory, n_faces):
    model = detector
    faces = []
    # enumerate
    for filename in listdir(directory):
        # load image
        pixels = decode_img(directory + filename)
        pixels = pixels.numpy()
        # get face
        face = extract_face(model, pixels)
        if face is None:
            continue
        # store
        faces.append(face)
        print(len(faces), face.shape)
        # stop
        if len(faces) >= n_faces:
            break
    faces = np.asarray(faces)
    print(faces.shape)
    return faces

# plot samples
def plot_faces(faces, n):
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis("off")
        plt.imshow(faces[i])
    plt.show()

# EXAMPLE

# directory = 'celeb_images/'

# faces = load_faces(directory, 5)
# print("Loaded {} faces as {}".format(len(faces), faces.shape))
# np.savez_compressed('img_align_celeba.npz', faces)

# # Load npz
# face_array = np.load('img_align_celeba.npz')
# print(face_array['arr_0'].shape)
#
# plot_faces(face_array['arr_0'], 4)
