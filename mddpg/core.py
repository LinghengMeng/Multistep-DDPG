import numpy as np
import tensorflow as tf


"""
Multilayer Perceptron
"""

class MLP(tf.keras.Model):
    def __init__(self, layer_sizes=[32],
                 hidden_activation=tf.keras.activations.relu, output_activation=tf.keras.activations.linear):
        super(MLP, self).__init__()
        self.layers_list = []
        for h_size in layer_sizes[:-1]:
            self.layers_list.append(tf.keras.layers.Dense(h_size, activation=hidden_activation))

        self.layers_list.append(tf.keras.layers.Dense(layer_sizes[-1], activation=output_activation))

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x