"""Functions to process data."""

import numpy as np


def compose (*functions):
    def composed(arg):
        for f in reversed(functions):
            arg = f(arg)
        return arg
    return composed


def force_to_0_1(x_data):
    return x_data / 255.0

    
def sphere_normalize(x_data):
  array = np.zeros(x_data.shape)
  for i in range(len(x_data)):
    array[i] = x_data[i] / np.sqrt(np.sum(np.square(x_data[i] * 1.0)))
  return array


def force_to_0_1_binary(x_data):
  return np.round(x_data / 255.0)


def label_filter(x_data, y_data, filter_func):
    zipped_lists = zip(x_data, y_data)
    filtered_data = [(x, y) for x, y in zipped_lists if filter_func(y)]
    filtered_x, filtered_y = zip(*filtered_data)
    filtered_x = np.array(filtered_x)
    if len(filtered_x.shape) == 3:
      filtered_x = np.expand_dims(filtered_x, axis=-1)
    return filtered_x, np.array(filtered_y)


def generate_factored(images, num_gen_images):
  num_images, frame_shape = images.shape[0], images.shape[1:]
  random_images = np.zeros((num_gen_images,) + frame_shape)
  for i in range(num_gen_images):
    for j in range(frame_shape[0]):
      for k in range(frame_shape[1]):
        idx = np.random.randint(0, num_images)
        random_images[i][j][k] = images[idx][j][k]
  return random_images
