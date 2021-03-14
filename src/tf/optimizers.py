import tensorflow as tf

def get_optimizer(name):
  if name == 'Adam':
    return tf.keras.optimizers.Adam()