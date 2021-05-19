import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Conv2D, MaxPool2D, Flatten, Dense

# Create simple model via functional API
def get_simple_cnn():
  model_input = Input(shape=(360, 360, 3), name="input_img")
  x = Conv2D(32, (3, 3), padding = "same", activation = "relu", name = "conv_1")(model_input)
  x = MaxPool2D(pool_size = (2, 2), name = "pool_1")(x)
  x = Conv2D(64, (3, 3), padding = "same", activation = "relu", name = "conv_2")(x)
  x = MaxPool2D(pool_size = (2, 2), name = "pool_2")(x)
  x = Conv2D(64, (3, 3), padding = "same", activation = "relu", name = "conv_3")(x)
  x = Flatten(name = "flat_1")(x)
  x = Dense(100, activation='relu', name = "dense_1")(x)
  x = Dense(20, activation='relu', name = "dense_2")(x)
  model_output = Dense(1, name = "dense_3")(x)
  return tf.keras.Model(model_input, model_output, name = "Simple_CNN")

# Create baseline model via functional API
def get_small_cnn():
  model_input = Input(shape=(360, 360, 3), name="input_img")
  x = Conv2D(16, (3, 3), padding = "same", activation = "relu", name = "conv_1")(model_input)
  x = MaxPool2D(pool_size = (4, 4), name = "pool_1")(x)
  x = Conv2D(32, (3, 3), padding = "same", activation = "relu", name = "conv_2")(x)
  x = MaxPool2D(pool_size = (4, 4), name = "pool_2")(x)
  x = Conv2D(32, (3, 3), padding = "same", activation = "relu", name = "conv_3")(x)
  x = Flatten(name = "flat_1")(x)
  x = Dense(100, activation='relu', name = "dense_1")(x)
  x = Dense(20, activation='relu', name = "dense_2")(x)
  model_output = Dense(1, name = "dense_3")(x)
  return tf.keras.Model(model_input, model_output, name = "Small_CNN")

# Create mini model via functional API
def get_mini_cnn():
  model_input = Input(shape=(360, 360, 3), name="input_img")
  x = Conv2D(8, (3, 3), padding = "same", activation = "relu", name = "conv_1")(model_input)
  x = MaxPool2D(pool_size = (8, 8), name = "pool_1")(x)
  x = Conv2D(16, (3, 3), padding = "same", activation = "relu", name = "conv_2")(x)
  x = MaxPool2D(pool_size = (8, 8), name = "pool_2")(x)
  x = Conv2D(16, (3, 3), padding = "same", activation = "relu", name = "conv_3")(x)
  x = Flatten(name = "flat_1")(x)
  x = Dense(100, activation='relu', name = "dense_1")(x)
  x = Dense(20, activation='relu', name = "dense_2")(x)
  model_output = Dense(1, name = "dense_3")(x)
  return tf.keras.Model(model_input, model_output, name = "Mini_CNN")

# Create simple model via functional API
def get_simple_cnn_class():
  model_input = Input(shape=(360, 360, 3), name="input_img")
  x = Conv2D(32, (3, 3), padding = "same", activation = "relu", name = "conv_1")(model_input)
  x = MaxPool2D(pool_size = (2, 2), name = "pool_1")(x)
  x = Conv2D(64, (3, 3), padding = "same", activation = "relu", name = "conv_2")(x)
  x = MaxPool2D(pool_size = (2, 2), name = "pool_2")(x)
  x = Conv2D(64, (3, 3), padding = "same", activation = "relu", name = "conv_3")(x)
  x = Flatten(name = "flat_1")(x)
  x = Dense(100, activation='relu', name = "dense_1")(x)
  x = Dense(20, activation='relu', name = "dense_2")(x)
  model_output = Dense(1, activation = "sigmoid", name = "dense_3")(x)
  return tf.keras.Model(model_input, model_output, name = "Simple_CNN")

# Create small model via functional API
def get_small_cnn_class():
  model_input = Input(shape=(360, 360, 3), name="input_img")
  x = Conv2D(16, (3, 3), padding = "same", activation = "relu", name = "conv_1")(model_input)
  x = MaxPool2D(pool_size = (4, 4), name = "pool_1")(x)
  x = Conv2D(32, (3, 3), padding = "same", activation = "relu", name = "conv_2")(x)
  x = MaxPool2D(pool_size = (4, 4), name = "pool_2")(x)
  x = Conv2D(32, (3, 3), padding = "same", activation = "relu", name = "conv_3")(x)
  x = Flatten(name = "flat_1")(x)
  x = Dense(100, activation='relu', name = "dense_1")(x)
  x = Dense(20, activation='relu', name = "dense_2")(x)
  model_output = Dense(1, activation = "sigmoid", name = "dense_3")(x)
  return tf.keras.Model(model_input, model_output, name = "Base_CNN")

# Create mini model via functional API
def get_mini_cnn():
  model_input = Input(shape=(360, 360, 3), name="input_img")
  x = Conv2D(8, (3, 3), padding = "same", activation = "relu", name = "conv_1")(model_input)
  x = MaxPool2D(pool_size = (8, 8), name = "pool_1")(x)
  x = Conv2D(16, (3, 3), padding = "same", activation = "relu", name = "conv_2")(x)
  x = MaxPool2D(pool_size = (8, 8), name = "pool_2")(x)
  x = Conv2D(16, (3, 3), padding = "same", activation = "relu", name = "conv_3")(x)
  x = Flatten(name = "flat_1")(x)
  x = Dense(100, activation='relu', name = "dense_1")(x)
  x = Dense(20, activation='relu', name = "dense_2")(x)
  model_output = Dense(1, activation = "sigmoid", name = "dense_3")(x)
  return tf.keras.Model(model_input, model_output, name = "Mini_CNN")

# Function to get model based on name
def get_model(name, task_type = "reg"):
  if task_type == "reg":
    if name == 'SimpleCNN':
      return get_simple_cnn()
    elif name == 'SmallCNN':
      return get_small_cnn()
    elif name == 'MiniCNN':
      return get_mini_cnn()
  elif task_type == "class":
    if name == 'SimpleCNN':
      return get_simple_cnn_class()
    elif name == 'SmallCNN':
      return get_small_cnn_class()
    elif name == 'MiniCNN':
      return get_mini_cnn_class()
  print("Invalid model or mode!")
  return None
