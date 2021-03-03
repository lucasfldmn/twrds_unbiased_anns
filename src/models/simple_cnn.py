import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Conv2D, MaxPool2D, Flatten, Dense


class SimpleCNN(tf.keras.Model):
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define layers
        self.network_layers = [
          # Convolutional layers with zero padding and maxpooling inbetween
          Conv2D(32, (3, 3), padding = "same", activation = "relu", input_shape=(360, 360, 4), name = "conv_1"), 
          MaxPool2D(pool_size = (2, 2), name = "pool_1"),          
          Conv2D(64, (3, 3), padding = "same", activation = "relu", name = "conv_2"),
          MaxPool2D(pool_size = (2, 2), name = "pool_2"),
          Conv2D(64, (3, 3), padding = "same", activation = "relu", name = "conv_3"),
          # Flatten the feature maps
          Flatten(name = "flat_1"),
          # Fully connected layers to funnel flat tensor into single value
          Dense(100, activation='relu', name = "dense_1"),
          Dense(20, activation='relu', name = "dense_2"),
          Dense(1, name = "dense_3")
        ]    
        
    def call(self, x):
        # Calculate forward step through all layers
        for layer in self.network_layers:
          x = layer(x)
        return x

# Create model via functional API
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

