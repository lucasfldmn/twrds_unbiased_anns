import numpy as np
import tensorflow as tf
import scipy.stats as stats
import random
from .shapes import make_circle, make_square


def draw_from_truncated_normal_distribution(n_samples, mean, stddev=20):
    # Set lower and upper bounds for truncation
    lower = 20
    upper = 150
    # Set parameters of normal distribution
    mu = mean
    sigma = stddev
    # Randomly sample
    samples = stats.truncnorm.rvs(
        (lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=n_samples)
    return np.reshape(samples.round(), (n_samples, 1))


def get_sample_params(category, m_diff, std, share):
    # Calculate group means
    mean_1 = 100
    mean_2 = 100 - m_diff
    # Calculate group share
    share_1 = (100 - share) / 2
    share_2 = share / 2
    # Set parameters based on the categories used to determine the split
    if category == "color":
        white_square = [share_1, mean_1, 20]
        white_circle = [share_1, mean_1, 20]
        colorful_square = [share_2, mean_2, std]
        colorful_circle = [share_2, mean_2, std]
    elif category == "shape":
        white_square = [share_1, mean_1, 20]
        white_circle = [share_2, mean_2, std]
        colorful_square = [share_1, mean_1, 20]
        colorful_circle = [share_2, mean_2, std]
    return white_square, white_circle, colorful_square, colorful_circle


def create_sample_array(n_samples, white_square, white_circle, colorful_square, colorful_circle):
    # Calculate number of samples for each group
    n_white_square = round(n_samples * white_square[0] / 100)
    n_white_circle = round(n_samples * white_circle[0] / 100)
    n_colorful_square = round(n_samples * colorful_square[0] / 100)
    n_colorful_circle = round(n_samples * colorful_circle[0] / 100)

    # White squares
    color = np.ones((n_white_square, 1), dtype=bool)  # True = white
    shape = np.ones((n_white_square, 1), dtype=bool)  # True = square
    size = draw_from_truncated_normal_distribution(
        n_white_square, mean=white_square[1], stddev=white_square[2])
    white_squares = np.hstack((color, shape, size))

    # White circles
    color = np.ones((n_white_circle, 1), dtype=bool)  # True = white
    shape = np.zeros((n_white_circle, 1), dtype=bool)  # False = circle
    size = draw_from_truncated_normal_distribution(
        n_white_circle, mean=white_circle[1], stddev=white_circle[2])
    white_circles = np.hstack((color, shape, size))

    # Colorful squares
    color = np.zeros((n_colorful_square, 1), dtype=bool)  # False = colorful
    shape = np.ones((n_colorful_square, 1), dtype=bool)  # True = square
    size = draw_from_truncated_normal_distribution(
        n_colorful_square, mean=colorful_square[1], stddev=colorful_square[2])
    colorful_squares = np.hstack((color, shape, size))

    # Colorful circles
    color = np.zeros((n_colorful_circle, 1), dtype=bool)  # False = colorful
    shape = np.zeros((n_colorful_circle, 1), dtype=bool)  # False = circle
    size = draw_from_truncated_normal_distribution(
        n_colorful_circle, mean=colorful_circle[1], stddev=colorful_circle[2])
    colorful_circles = np.hstack((color, shape, size))

    # Stack all together
    samples = np.vstack((white_squares, white_circles,
                         colorful_squares, colorful_circles))

    # Shuffle array
    np.random.shuffle(samples)

    # Return result
    return samples


def convert_sample_to_np_array(sample, colors):
    # Get sample color
    if sample[0]:
        sample_color = 'white'
        color_bin = 0
    else:
        sample_color = np.random.choice(colors)
        color_bin = 1

    # Get size of sample
    sample_size = int(sample[2])

    # Call shape generator based on sample shape
    if sample[1]:
        shape_array = make_square(color=sample_color, size=sample_size)
        shape_bin = 0
    else:
        shape_array = make_circle(color=sample_color, size=sample_size)
        shape_bin = 1

    # Return numpy array and size
    return shape_array, sample_size, color_bin, shape_bin


def convert_sample_to_tensor(sample, colors):
    # Convert sample to numpy array
    sample_np_array, sample_size, color_bin, shape_bin = convert_sample_to_np_array(sample, colors)
    # Convert array to tensor
    img_tensor = tf.convert_to_tensor(sample_np_array, dtype=tf.int32)
    # Divide image tensor by 255 to normalize values
    img_tensor = img_tensor / 255
    # Return tensor and size
    return img_tensor, sample_size


def get_sample_data(samples):
    # Iterate over samples and create list of numpy image arrays and list of target
    images = []
    targets = []
    colors = []
    shapes = []
    for sample in samples:
        # Convert sample to numpy array of the image
        shape_tensor, sample_size, color_bin, shape_bin = convert_sample_to_np_array(sample, colors)
        images.append(shape_tensor)
        targets.append(sample_size)
        colors.append(color_bin)
        shapes.append(shape_bin)

    # Convert both to tensors
    img_tensor = tf.convert_to_tensor(images, dtype=tf.int32)
    target_tensor = tf.convert_to_tensor(targets, dtype=tf.float32)
    color_tensor = tf.convert_to_tensor(colors, dtype=tf.int32)
    shape_tensor = tf.convert_to_tensor(shapes, dtype=tf.int32) 

    # Divide image tensor by 255 to normalize values
    img_tensor = img_tensor / 255

    # Return images and targets
    return img_tensor, (target_tensor, color_tensor, shape_tensor)


def gen_from_sample(samples, colors, attributes):
    # Iterate over samples and create list of numpy image arrays and list of target
    for sample in samples:
        # Convert sample to numpy array of the image
        shape_np, sample_size, color_bin, shape_bin = convert_sample_to_np_array(sample, colors)
        # Convert both to tensors
        img_tensor = tf.convert_to_tensor(shape_np, dtype=tf.float32)
        target_tensor = tf.convert_to_tensor(sample_size, dtype=tf.float32)
        color_tensor = tf.convert_to_tensor(color_bin, dtype=tf.int32)
        shape_tensor = tf.convert_to_tensor(shape_bin, dtype=tf.int32) 
        # Divide image tensor by 255 to normalize values
        img_tensor = img_tensor / 255
        # Yield input and targets + attributes
        if "color" in attributes:
            if "shape" in attributes:
                yield img_tensor, (target_tensor, color_tensor, shape_tensor)
            else:
                yield img_tensor, (target_tensor, color_tensor)
        else:
            if "shape" in attributes:
                yield img_tensor, (target_tensor, shape_tensor)
            else:
                yield img_tensor, target_tensor

def gen_from_sample_class(samples, colors, threshold, noise, distractor, attributes):
    # Iterate over samples and create list of numpy image arrays and list of target
    for sample in samples:
        # Convert sample to numpy array of the image
        shape_np, sample_size, color_bin, shape_bin = convert_sample_to_np_array(sample, colors)
        # Convert img to tensor
        img_tensor = tf.convert_to_tensor(shape_np, dtype=tf.float32)
        # Get label based on shape size, threshold and group
        if distractor == "color":
            # Check if sample belongs to 'over' group -> fixed threshold
            if color_bin == 0:
                label = get_label(sample_size, 75, noise)
            else:
                label = get_label(sample_size, threshold, noise)
        else:
             # Check if sample belongs to 'over' group -> fixed threshold
            if shape_bin == 0:
                label = get_label(sample_size, 75, noise)
            else:
                label = get_label(sample_size, threshold, noise)        
        target_tensor = tf.convert_to_tensor(label, dtype=tf.int32)
        # Get targets for attributes
        color_tensor = tf.convert_to_tensor(color_bin, dtype=tf.int32)
        shape_tensor = tf.convert_to_tensor(shape_bin, dtype=tf.int32)
        # Divide image tensor by 255 to normalize values
        img_tensor = img_tensor / 255
        # Yield input and targets + attributes
        if "color" in attributes:
            if "shape" in attributes:
                yield img_tensor, (target_tensor, color_tensor, shape_tensor)
            else:
                yield img_tensor, (target_tensor, color_tensor)
        else:
            if "shape" in attributes:
                yield img_tensor, (target_tensor, shape_tensor)
            else:
                yield img_tensor, target_tensor        

def get_label(size, threshold, noise):
    # Check if size is above threshold and assign label
    label = 1 if size >= threshold else 0
    # Randomly flip label based on noise
    if random.random() <= noise/100:
        label = 0 if label == 1 else 1
    return label

def dataset_from_gen(sample, n_epochs, batch_size, colors, task_type = "reg", threshold = 75, noise = 0, distractor = 'color', attributes = ["color", "shape"]):
    if task_type == "reg":
        if len(attributes) == 2:
            dataset = tf.data.Dataset.from_generator(
                lambda: gen_from_sample(sample, colors, attributes),
                output_signature=(
                    tf.TensorSpec(shape=(360, 360, 3), dtype=tf.float32),
                    (tf.TensorSpec(shape=(), dtype=tf.float32), 
                    tf.TensorSpec(shape=(), dtype=tf.int32), 
                    tf.TensorSpec(shape=(), dtype=tf.int32))
                )
            )
        elif len(attributes) == 1:
            dataset = tf.data.Dataset.from_generator(
                lambda: gen_from_sample(sample, colors, attributes),
                output_signature=(
                    tf.TensorSpec(shape=(360, 360, 3), dtype=tf.float32),
                    (tf.TensorSpec(shape=(), dtype=tf.float32), 
                    tf.TensorSpec(shape=(), dtype=tf.int32))
                )
            )
        else:
            dataset = tf.data.Dataset.from_generator(
                lambda: gen_from_sample(sample, colors, attributes),
                output_signature=(
                    tf.TensorSpec(shape=(360, 360, 3), dtype=tf.float32),
                    (tf.TensorSpec(shape=(), dtype=tf.float32))
                )
            )
    elif task_type == "class":
        if len(attributes) == 2:
            dataset = tf.data.Dataset.from_generator(
                lambda: gen_from_sample_class(sample, colors, threshold, noise, distractor, attributes),
                output_signature=(
                    tf.TensorSpec(shape=(360, 360, 3), dtype=tf.float32),
                    (tf.TensorSpec(shape=(), dtype=tf.float32), 
                    tf.TensorSpec(shape=(), dtype=tf.int32), 
                    tf.TensorSpec(shape=(), dtype=tf.int32))
                )
            )
        elif len(attributes) == 1:
            dataset = tf.data.Dataset.from_generator(
                lambda: gen_from_sample_class(sample, colors, threshold, noise, distractor, attributes),
                output_signature=(
                    tf.TensorSpec(shape=(360, 360, 3), dtype=tf.float32),
                    (tf.TensorSpec(shape=(), dtype=tf.float32), 
                    tf.TensorSpec(shape=(), dtype=tf.int32))
                )
            )
        else:
            dataset = tf.data.Dataset.from_generator(
                lambda: gen_from_sample_class(sample, colors, threshold, noise, distractor, attributes),
                output_signature=(
                    tf.TensorSpec(shape=(360, 360, 3), dtype=tf.float32),
                    (tf.TensorSpec(shape=(), dtype=tf.float32))
                )
            )        
    dataset = dataset.repeat(n_epochs)
    dataset = dataset.batch(batch_size)
    return dataset
