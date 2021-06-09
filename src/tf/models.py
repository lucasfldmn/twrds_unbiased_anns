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
  x = Dense(100, activation="relu", name = "dense_1")(x)
  x = Dense(20, activation="relu", name = "dense_2")(x)
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
  x = Dense(100, activation="relu", name = "dense_1")(x)
  x = Dense(20, activation="relu", name = "dense_2")(x)
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
  x = Dense(100, activation="relu", name = "dense_1")(x)
  x = Dense(20, activation="relu", name = "dense_2")(x)
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
  x = Dense(100, activation="relu", name = "dense_1")(x)
  x = Dense(20, activation="relu", name = "dense_2")(x)
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
  x = Dense(100, activation="relu", name = "dense_1")(x)
  x = Dense(20, activation="relu", name = "dense_2")(x)
  model_output = Dense(1, activation = "sigmoid", name = "dense_3")(x)
  return tf.keras.Model(model_input, model_output, name = "Base_CNN")

# Create mini model via functional API
def get_mini_cnn_class():
  model_input = Input(shape=(360, 360, 3), name="input_img")
  x = Conv2D(8, (3, 3), padding = "same", activation = "relu", name = "conv_1")(model_input)
  x = MaxPool2D(pool_size = (8, 8), name = "pool_1")(x)
  x = Conv2D(16, (3, 3), padding = "same", activation = "relu", name = "conv_2")(x)
  x = MaxPool2D(pool_size = (8, 8), name = "pool_2")(x)
  x = Conv2D(16, (3, 3), padding = "same", activation = "relu", name = "conv_3")(x)
  x = Flatten(name = "flat_1")(x)
  x = Dense(100, activation="relu", name = "dense_1")(x)
  x = Dense(20, activation="relu", name = "dense_2")(x)
  model_output = Dense(1, activation = "sigmoid", name = "dense_3")(x)
  return tf.keras.Model(model_input, model_output, name = "Mini_CNN")

# Gradient reversal operation
@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad

# Layer that reverses the gradient
class GradReverse(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return grad_reverse(x)

def get_grad_model(n_attributes, attr_loss_weight, target_loss_weight = 1., classification = False):    
    # Input
    model_input = Input(shape=(360, 360, 3), name="input_img")
    # Feature extractor
    x = Conv2D(32, (3, 3), padding = "same", activation = "relu", name = "conv_1")(model_input)
    x = MaxPool2D(pool_size = (2, 2), name = "pool_1")(x)
    x = Conv2D(64, (3, 3), padding = "same", activation = "relu", name = "conv_2")(x)
    x = MaxPool2D(pool_size = (2, 2), name = "pool_2")(x)
    x = Conv2D(64, (3, 3), padding = "same", activation = "relu", name = "conv_3")(x)
    x = Flatten(name = "flat_1")(x)
    # Target branch
    x_target = Dense(100, activation="relu", name = "target_dense_1")(x)
    x_target = Dense(20, activation="relu", name = "target_dense_2")(x_target)
    if classification:
        target_output = Dense(1, activation = "sigmoid", name = "target_output")(x_target)
    else:
        target_output = Dense(1, name = "target_output")(x_target)
    model_outputs = [target_output]    
    # Create losses, weights and metrics
    if classification:
        losses = {"target_output": tf.keras.losses.BinaryCrossentropy()}
        target_metric = "accuracy"
    else: 
        losses = {"target_output": tf.keras.losses.MeanSquaredError()}
        target_metric = "mean_squared_error"
    weights = {"target_output": target_loss_weight}
    metrics = {"target_output": target_metric}    
    # Split attribute loss over branches
    attr_branch_weight = attr_loss_weight / n_attributes    
    # Attribute branches
    for i in range(n_attributes):
        # Gradient reversal layer
        x_attr = GradReverse()(x)
        # Funnel into sigmoid for binary classification of attributes
        x_attr = Dense(100, activation="relu", name = "attr_{}_dense_1".format(i))(x_attr)
        x_attr = Dense(20, activation="relu", name = "attr_{}_dense_2".format(i))(x_attr)
        output_name = "attr_{}_output".format(i)
        attr_output = Dense(1, activation = "sigmoid", name = output_name)(x_attr) 
        # Add to outputs
        model_outputs.append(attr_output)
        # Add to losses, weights and metrics
        losses[output_name] = tf.keras.losses.BinaryCrossentropy()
        weights[output_name] = attr_branch_weight
        metrics[output_name] = "accuracy"        
    # Make model
    model = tf.keras.Model(inputs = model_input, outputs = model_outputs, name = "GRAD_CNN")       
    # Compile model
    model.compile(optimizer = tf.keras.optimizers.Adam(), loss = losses, loss_weights = weights, metrics = metrics)    
    # Return finished model
    return model

# Function to get model based on name
def get_model(name, task_type = "reg", attr_loss_weight = 1000, target_loss_weight = 1., n_attributes = 2, loss = tf.keras.losses.MeanSquaredError()):
  if task_type == "reg":
    if name == "SimpleCNN":
      return get_simple_cnn()
    elif name == "SmallCNN":
      return get_small_cnn()
    elif name == "MiniCNN":
      return get_mini_cnn()
    elif name == "GRAD":
      return get_grad_model(n_attributes, attr_loss_weight, target_loss_weight, False)
  elif task_type == "class":
    if name == "SimpleCNN":
      return get_simple_cnn_class()
    elif name == "SmallCNN":
      return get_small_cnn_class()
    elif name == "MiniCNN":
      return get_mini_cnn_class()
    elif name == "GRAD":
      return get_grad_model(n_attributes, attr_loss_weight, target_loss_weight, True)
  print("Invalid model or mode!")
  return None
