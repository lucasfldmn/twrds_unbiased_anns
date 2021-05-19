import matplotlib.colors as mcolors
import numpy as np
import cv2
from skimage.util import random_noise

def make_shape(shape = 'square', color = 'red', size = 50):
    # Check if color is given as string or RGB
    if isinstance(color, str):
        # Convert color string to RGB
        (r, b, g) = mcolors.to_rgb(color)
        (r, b, g) = (r*255, b*255, g*255)
    else: 
        # Extract rgb values
        (r, b, g) = color
    # Make empty image
    img = np.zeros((360, 360, 3), dtype = "uint8")
    # Draw shape depending on size
    if shape == 'square':
        # Calculate upper left image edge based on size
        starting_point = 180-size
        end_point = starting_point + size*2
        cv2.rectangle(img, (starting_point, starting_point), (end_point, end_point), (r, b, g), -1) 
    elif shape == 'circle':
        cv2.circle(img, (180, 180), size, (r, g, b), -1) 
    # Add salt-and-pepper noise to the image.
    noise_img = random_noise(img, mode='s&p',amount=0.1) 
    noise_img = np.array(255*noise_img, dtype = 'uint8')
    # Return image array
    return noise_img

def make_square(color = 'red', size = 50):
	return make_shape('square', color, size)

def make_circle(color = 'red', size = 50):
	return make_shape('circle', color, size)