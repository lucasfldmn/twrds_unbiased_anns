import tensorflow as tf

def get_loss(name):
  if name == 'mse':
    return tf.keras.losses.MeanSquaredError()
  elif name == 'bce':
    return tf.keras.losses.BinaryCrossentropy()
  elif name == 'custom': # TODO custom loss functions
    return None

