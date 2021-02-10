import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, Flatten, Dense


class SimpleCNN(tf.keras.Model):
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define layers
        self.network_layers = [
          # Convolutional layers with zero padding and maxpooling inbetween
          Conv2D(32, (3, 3), padding = "same", activation = "relu", input_shape=(360, 360, 4)), 
          MaxPool2D(pool_size = (2, 2)),          
          Conv2D(64, (3, 3), padding = "same", activation = "relu"),
          MaxPool2D(pool_size = (2, 2)),
          Conv2D(64, (3, 3), padding = "same", activation = "relu"),
          # Flatten the feature maps
          Flatten(),
          # Fully connected layers to funnel flat tensor into single value
          Dense(100, activation='relu'),
          Dense(20, activation='relu'),
          Dense(1)
        ]    
        
    def call(self, x):
        # Calculate forward step through all layers
        for layer in self.network_layers:
          x = layer(x)
        return x