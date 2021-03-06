import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from .samples import draw_from_truncated_normal_distribution, convert_sample_to_tensor, get_label

def create_eval_samples(n_samples, same_data = True):  
  if same_data:

    # Set parameters
    percentage = 25
    mean = 100

    # Calculate number of samples for each group
    n_samples_per_cat = round(n_samples * percentage / 100)    

    # Get size data (actuals)
    size = draw_from_truncated_normal_distribution(n_samples_per_cat, mean, 20)

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

def load_eval_samples(eval_sample_filepath):
  # Load evaluation sample
  with open(eval_sample_filepath, 'rb') as filehandle:
    eval_samples = pickle.load(filehandle)
  return eval_samples

def evaluate_performance(group_sample, model, colors):
  # Feed sample to model and store targets and prediction
  actual = []
  target_prediction = []
  color_prediction = []
  shape_prediction = []
  for single_sample in group_sample:
    # Convert to tensor
    shape_tensor, target_size = convert_sample_to_tensor(single_sample, colors)
    # Reshape the tensor
    shape_tensor = tf.reshape(shape_tensor, [1,360,360,3])
    # Feed to model
    output = model(shape_tensor)
    # Store prediction and target in list
    actual.append(target_size)
    # Check if predictions are list -> GRAD model
    if isinstance(output, list):
      target_prediction.append(output[0].numpy()[0][0])
      if len(output) > 1:
        color_prediction.append(output[1].numpy()[0][0])
      if len(output) > 2:
        shape_prediction.append(output[2].numpy()[0][0])
    else:
      target_prediction.append(output.numpy()[0][0])
  # Return lists of actual values and predicted values
  return actual, target_prediction, color_prediction, shape_prediction

def evaluate_performance_class(group_sample, model, colors, threshold):
  # Feed sample to model and store size, actual and prediction
  size = []
  actual = []
  prediction = []
  for single_sample in group_sample:
    # Convert to tensor
    shape_tensor, target_size = convert_sample_to_tensor(single_sample, colors)
    # Reshape the tensor
    shape_tensor = tf.reshape(shape_tensor, [1,360,360,3])
    # Feed to model
    output = model(shape_tensor)
    # Get true label
    label = get_label(target_size, threshold, noise = 0)
    # Store size, prediction and target in list
    size.append(target_size)
    actual.append(label)
    prediction.append(output.numpy()[0][0])
  # Return lists of actual values and predicted values
  return size, actual, prediction

def evaluate_model(model, eval_samples, row, results, colors, task_type = "reg", threshold = 75):
  # Evaluate performance depending on task type
  if task_type == "reg":
    # Go through evaluation samples and create a row for each
    for (group_sample, (label)) in eval_samples:    
      actuals, target_predictions, color_predictions, shape_predictions = evaluate_performance(group_sample, model, colors)
      # Store results      
      row["shape_color"] = label.split("_")[0]
      row["shape_type"] = label.split("_")[1]
      # Write everything to results
      for idx, actual in enumerate(actuals):
        row["actual"] = actual
        row["prediction"] = target_predictions[idx]
        if len(color_predictions) > 0:
          row["color_prediction"] = color_predictions[idx]
        if len(shape_predictions) > 0:
          row["shape_prediction"] = shape_predictions[idx]
        results.append(row.copy())
  elif task_type == "class":
    # Go through evaluation samples and create a row for each
    for (group_sample, (label)) in eval_samples:
      sizes, actuals, predictions = evaluate_performance_class(group_sample, model, colors, threshold)
      # Store results      
      row["shape_color"] = label.split("_")[0]
      row["shape_type"] = label.split("_")[1]
      # Write everything to results
      for idx, actual in enumerate(actuals):
        row["size"] = sizes[idx]
        row["actual"] = actual
        row["prediction"] = predictions[idx]
        results.append(row.copy())

def store_results(results, filename):
  # Create dataframe from results
  result_df = pd.DataFrame(results)
  # Write dataframe to excel
  result_df.to_excel(filename)