import numpy as np
import tensorflow as tf
import scipy.stats as stats
from .shapes import make_circle, make_square

def draw_from_truncated_normal_distribution(n_samples, mean, stddev = 30):
  # Set lower and upper bounds for truncation
  lower = 20
  upper = 150
  # Set parameters of normal distribution
  mu = mean
  sigma = stddev
  # Randomly sample
  samples = stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc = mu, scale = sigma, size = n_samples)
  return np.reshape(samples.round(), (n_samples, 1))

def create_sample_array(n_samples, white_square, white_circle, colorful_square, colorful_circle):
  # Calculate number of samples for each group
  n_white_square = round(n_samples * white_square[0] / 100)
  n_white_circle = round(n_samples * white_circle[0] / 100)
  n_colorful_square = round(n_samples * colorful_square[0] / 100)
  n_colorful_circle = round(n_samples * colorful_circle[0] / 100)

  # White squares
  color = np.ones((n_white_square,1), dtype = bool) # True = white
  shape = np.ones((n_white_square,1), dtype = bool) # True = square
  size = draw_from_truncated_normal_distribution(n_white_square, mean = white_square[1], stddev = white_square[2])
  white_squares = np.hstack((color, shape, size))

  # White circles
  color = np.ones((n_white_circle,1), dtype = bool) # True = white
  shape = np.zeros((n_white_circle,1), dtype = bool) # False = circle
  size = draw_from_truncated_normal_distribution(n_white_circle, mean = white_circle[1], stddev = white_circle[2])
  white_circles = np.hstack((color, shape, size))

  # Colorful squares
  color = np.zeros((n_colorful_square,1), dtype = bool) # False = colorful
  shape = np.ones((n_colorful_square,1), dtype = bool) # True = square
  size = draw_from_truncated_normal_distribution(n_colorful_square, mean = colorful_square[1], stddev = colorful_square[2])
  colorful_squares = np.hstack((color, shape, size))

  # Colorful circles
  color = np.zeros((n_colorful_circle,1), dtype = bool) # False = colorful
  shape = np.zeros((n_colorful_circle,1), dtype = bool) # False = circle
  size = draw_from_truncated_normal_distribution(n_colorful_circle, mean = colorful_circle[1], stddev = colorful_circle[2])
  colorful_circles = np.hstack((color, shape, size))

  # Stack all together
  samples = np.vstack((white_squares, white_circles, colorful_squares, colorful_circles))

  # Shuffle array
  np.random.shuffle(samples)

  # Return result
  return samples

def convert_sample_to_np_array(sample, colors):  
  # Get sample color
  if sample[0]:
    sample_color = 'white'
  else:
    sample_color = np.random.choice(colors)

  # Get size of sample
  sample_size = int(sample[2])

  # Call shape generator based on sample shape
  if sample[1]:
    shape_array = make_square(color = sample_color, size = sample_size)
  else:
    shape_array = make_circle(color = sample_color, size = sample_size)

  # Return numpy array and size
  return shape_array, sample_size

def convert_sample_to_tensor(sample, colors):
  # Convert sample to numpy array
  sample_np_array, sample_size = convert_sample_to_np_array(sample, colors)
  # Convert array to tensor
  img_tensor = tf.convert_to_tensor(sample_np_array, dtype=tf.int32)
  # Divide image tensor by 255 to normalize values
  img_tensor = img_tensor / 255
  # Return tensor and size
  return img_tensor, sample_size

def make_tf_dataset(samples, divide_by_255 = True, colors = ['red']):
  
  # Iterate over samples and create list of numpy image arrays and list of target
  images = []
  targets = []
  for sample in samples:
    # Convert sample to numpy array of the image
    shape_tensor, sample_size = convert_sample_to_np_array(sample, colors)
    images.append(shape_tensor)
    targets.append(sample_size)

  # Convert both to tensors
  img_tensor = tf.convert_to_tensor(images, dtype=tf.int32)
  target_tensor = tf.convert_to_tensor(targets, dtype=tf.float32)

  # Divide image tensor by 255 to normalize values
  img_tensor = img_tensor / 255
  
  # Convert to tensorflow dataset and return results
  dataset = tf.data.Dataset.from_tensor_slices((img_tensor, target_tensor))
  return dataset