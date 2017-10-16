"""
# Assume features is of size [N, H, W, C] (batch_size, height, width, channels).
# Transpose it to [N, C, H, W], then reshape to [N * C, H * W] to compute softmax
# jointly over the image dimensions.
features = tf.reshape(tf.transpose(features, [0, 3, 1, 2]), [N * C, H * W])
softmax = tf.nn.softmax(features)
# Reshape and transpose back to original format.
softmax = tf.transpose(tf.reshape(softmax, [N, C, H, W]), [0, 2, 3, 1])

...

# Assume that image_coords is a tensor of size [H, W, 2] representing the image
# coordinates of each pixel.
# Convert softmax to shape [N, H, W, C, 1]
softmax = tf.expand_dims(softmax, -1)
# Convert image coords to shape [H, W, 1, 2]
image_coords = tf.expand_dims(image_coords, 2)
# Multiply (with broadcasting) and reduce over image dimensions to get the result
# of shape [N, C, 2]
spatial_soft_argmax = tf.reduce_sum(softmax * image_coords, reduction_indices=[1, 2])
"""
from tensorflow.contrib.layers.python.layers.layers import spatial_softmax
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, features):
        return spatial_softmax(features)
        #features = K.reshape(K.transpose(x, [0, 3, 1, 2]), [self.N * self.C, self.H * self.W])
        #softmax = K.softmax(features)
        #softmax = K.transpose(K.reshape(softmax, [self.N, self.C, self.H, self.W]), [0, 2, 3, 1])
        #softmax = K.expand_dims(softmax, -1)
        #image_coords = K.expand_dims(image_coords, 2)
        #spatial_soft_argmax = K.reduce_sum(softmax * image_coords, reduction_indices=[1, 2])
        #return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        #[batch_size, num_channels * 2]
        return (input_shape[0], self.output_dim)