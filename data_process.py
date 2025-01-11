import os
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


batch_size = 32 # Minibatch size
buffer_size= 512 # Sample mini-batches from a subset of the dataset

images = np.load('./dataset/images.npy', mmap_mode='r')
file_paths = np.load('./dataset/file_paths.npy')

def read_image(path):
  """
    Return a 2D tensor reprensenting the pixels of the image located at <path>
  """
  img_shape=(512, 512)
  path_str = path.numpy().decode('UTF-8')
  image_data = images[np.where(file_paths == path_str)].reshape(img_shape)
  fg_mask = (image_data > (10 ** 2)).astype(int)
  image_data *= fg_mask
  tensor = tf.convert_to_tensor(image_data)
  return tensor


def convert_to_onehot(ds):
  """
    Convert each scalar valued target in ds to a vector of shape
    [num_classes, 1]
  """
  num_classes = 3
  I = tf.eye(num_classes, dtype=tf.int16)

  ds = ds.map(lambda x, y: (x, tf.py_function(lambda y: I[y.numpy() - 1], [y], [tf.int16])))
  ds = ds.map(lambda x, y: (x, tf.reshape(y, [3])))
  return ds


### Augmentation Operations ###

def add_noise(image):
  image = tf.cast(image, tf.float32)
  noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=1, dtype=tf.float32)
  noisy_img = image + noise
  return noisy_img

def rotate90(img):
  return tf.image.rot90(img)

def flip_vertical(img):
  return tf.image.flip_up_down(img)


def prepare(ds, shuffle=False, augment=False):
  """
    Prepare the dataset by:
      1) Applying a function to the dataset that will load and store <batch_size>
         images in memory at a time, to avoid exceeding RAM quota
      2) If augment=True, the dataset is augmented by performing a 90 deg
         rotation of each image and a flip along the horizontal axis.
  """

  # load images from disk only when needed
  ds = ds.map(lambda x, y: (tf.py_function(read_image, [x], [tf.int16]), y))
  ds = ds.map(lambda x, y: (tf.reshape(x, (512, 512, 1)), y))
  ds = convert_to_onehot(ds)

  orig_cardinality = ds.cardinality().numpy()

  if augment:
    ds_rot = ds.map(lambda x, y: (rotate90(x), y))
    ds_flip = ds.map(lambda x, y: (flip_vertical(x), y))
    ds = ds.concatenate(ds_rot)
    ds = ds.concatenate(ds_flip)

  if shuffle:
    ds = ds.shuffle(buffer_size)

  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size)
  return ds