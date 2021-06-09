import json
import tensorflow as tf
import gc
import os
import shutil

def load_configs_from_file(filepath):
  with open(filepath, 'r') as filehandle:
    config_json = json.load(filehandle)
  eval_sample_filename = config_json["general"]["eval_sample_filename"]
  repeats_per_model = config_json["general"]["repeats_per_model"]
  configs = config_json["configs"]
  return configs, eval_sample_filename, repeats_per_model

def free_memory():
  # Clear keras session
  tf.keras.backend.clear_session()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
  # Do garbage collection explicitly 
  gc.collect()

def load_config_from_file(filepath, task_type = "reg"):
  # Open json file
  with open(filepath, 'r') as filehandle:
    config_json = json.load(filehandle)
  # Load all variables
  name = config_json["name"]
  eval_sample_filename = config_json["eval_sample_filename"]
  dataset_size = config_json["dataset_size"]
  colors = config_json["colors"]
  optimizer = config_json["optimizer"]
  repeats_per_model = config_json["repeats_per_model"]
  batch_size = config_json["batch_size"]
  n_epochs = config_json["n_epochs"]
  mean_diffs = config_json["mean_diffs"]
  stddevs = config_json["stddevs"]
  minority_shares = config_json["minority_shares"]
  categorical = config_json["categorical"]
  models = config_json["models"]
  loss_functions = config_json["loss_functions"]
  
  # Load additional variables for classification task
  if task_type == "class":
    thresholds = config_json["thresholds"]
    noises = config_json["noises"]
    # Load attribute and target loss weights if they are there
    if "attr_loss_weight" in config_json.keys():
      attr_loss_weights = config_json["attr_loss_weight"]
      target_loss_weights = config_json["target_loss_weight"]
      n_attributes = config_json["n_attributes"]
      # Return variables
      return name, eval_sample_filename, dataset_size, colors, optimizer, repeats_per_model, batch_size, n_epochs, mean_diffs, stddevs, minority_shares, categorical, models, loss_functions, thresholds, noises, target_loss_weights, attr_loss_weights, n_attributes
    else:
      # Return variables
      return name, eval_sample_filename, dataset_size, colors, optimizer, repeats_per_model, batch_size, n_epochs, mean_diffs, stddevs, minority_shares, categorical, models, loss_functions, thresholds, noises
  else:
    # Load attribute and target loss weights if they are there
    if "attr_loss_weight" in config_json.keys():
      attr_loss_weights = config_json["attr_loss_weight"]
      target_loss_weights = config_json["target_loss_weight"]
      n_attributes = config_json["n_attributes"]
      # Return variables
      return name, eval_sample_filename, dataset_size, colors, optimizer, repeats_per_model, batch_size, n_epochs, mean_diffs, stddevs, minority_shares, categorical, models, loss_functions, target_loss_weights, attr_loss_weights, n_attributes
    else:
      # Return variables
      return name, eval_sample_filename, dataset_size, colors, optimizer, repeats_per_model, batch_size, n_epochs, mean_diffs, stddevs, minority_shares, categorical, models, loss_functions
  

def rmdir(path):
  shutil.rmtree(path)

def mkdir(path, delete = True):
  try: 
    os.mkdir(path) 
  except OSError as error: 
    print(error)
    if delete:
      print("Deleting directory and creating again ...")
      rmdir(path)
      os.mkdir(path)