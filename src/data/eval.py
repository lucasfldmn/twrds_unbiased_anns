import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from .samples import draw_from_truncated_normal_distribution, convert_sample_to_tensor

def create_eval_samples(n_samples, same_data = True):  
  if same_data:

    # Set parameters
    percentage = 25
    mean = 75

    # Calculate number of samples for each group
    n_samples_per_cat = round(n_samples * percentage / 100)    

    # Get size data (actuals)
    size = draw_from_truncated_normal_distribution(n_samples_per_cat, mean)

    # Build arrays for each category
    color = np.ones((n_samples_per_cat,1), dtype = bool) # True = white
    shape = np.ones((n_samples_per_cat,1), dtype = bool) # True = square
    white_squares = np.hstack((color, shape, size))

    color = np.ones((n_samples_per_cat,1), dtype = bool) # True = white
    shape = np.zeros((n_samples_per_cat,1), dtype = bool) # False = circle    
    white_circles = np.hstack((color, shape, size))

    # Colorful squares
    color = np.zeros((n_samples_per_cat,1), dtype = bool) # False = colorful
    shape = np.ones((n_samples_per_cat,1), dtype = bool) # True = square    
    colorful_squares = np.hstack((color, shape, size))

    # Colorful circles
    color = np.zeros((n_samples_per_cat,1), dtype = bool) # False = colorful
    shape = np.zeros((n_samples_per_cat,1), dtype = bool) # False = circle
    colorful_circles = np.hstack((color, shape, size))

  else:

    # Percentage of samples for each group
    perc_white_square = 25
    perc_white_circle = 25
    perc_colorful_square = 25
    perc_colorful_circle = 25

    # Means of normal distribution for the four groups
    mean_white_square = 75
    mean_white_circle = 75
    mean_colorful_square = 75
    mean_colorful_circle = 75  

    # Calculate number of samples for each group
    n_white_square = round(n_samples * perc_white_square / 100)
    n_white_circle = round(n_samples * perc_white_circle / 100)
    n_colorful_square = round(n_samples * perc_colorful_square / 100)
    n_colorful_circle = round(n_samples * perc_colorful_circle / 100)

    # White squares
    color = np.ones((n_white_square,1), dtype = bool) # True = white
    shape = np.ones((n_white_square,1), dtype = bool) # True = square
    size = draw_from_truncated_normal_distribution(n_white_square, mean_white_square)
    white_squares = np.hstack((color, shape, size))

    # White circles
    color = np.ones((n_white_circle,1), dtype = bool) # True = white
    shape = np.zeros((n_white_circle,1), dtype = bool) # False = circle
    size = draw_from_truncated_normal_distribution(n_white_circle, mean_white_circle)
    white_circles = np.hstack((color, shape, size))

    # Colorful squares
    color = np.zeros((n_colorful_square,1), dtype = bool) # False = colorful
    shape = np.ones((n_colorful_square,1), dtype = bool) # True = square
    size = draw_from_truncated_normal_distribution(n_colorful_square, mean_colorful_square)
    colorful_squares = np.hstack((color, shape, size))

    # Colorful circles
    color = np.zeros((n_colorful_circle,1), dtype = bool) # False = colorful
    shape = np.zeros((n_colorful_circle,1), dtype = bool) # False = circle
    size = draw_from_truncated_normal_distribution(n_colorful_circle, mean_colorful_circle)
    colorful_circles = np.hstack((color, shape, size))

  # Create labeled list of groups 
  samples = list(zip([white_squares, white_circles, colorful_squares, colorful_circles], ["white_square", "white_circle", "colorful_square", "colorful_circle"]))
  return samples

def create_and_save_eval_sample(n_eval_samples, filepath):
  # Create evaluation sample and save it
  eval_samples = create_eval_samples(n_eval_samples)
  with open(filepath, 'wb') as filehandle:
      pickle.dump(eval_samples, filehandle)

def load_eval_samples(eval_sample_filename):
  # Load evaluation sample
  with open("/content/twrds_unbiased_anns/data/eval/" + eval_sample_filename, 'rb') as filehandle:
    eval_samples = pickle.load(filehandle)
  return eval_samples

def evaluate_performance(group_sample, model, colors):
  # Feed sample to model and store targets and prediction
  actual = []
  prediction = []
  for single_sample in group_sample:
    # Convert to tensor
    shape_tensor, target_size = convert_sample_to_tensor(single_sample, colors)
    # Reshape the tensor
    shape_tensor = tf.reshape(shape_tensor, [1,360,360,3])
    # Feed to model
    output = model(shape_tensor)
    # Store prediction and target in list
    actual.append(target_size)
    prediction.append(output.numpy()[0][0])

  # Return lists of actual values and predicted values
  return actual, prediction

def store_results(results, filename):
  # Create dataframe from results
  result_df = pd.DataFrame(results)
  # Write dataframe to excel
  result_df.to_excel(filename)