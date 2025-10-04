import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

@keras.utils.register_keras_serializable()
class GraphNeuralOperator(layers.Layer):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.dense = layers.Dense(out_dim, kernel_initializer="he_normal")

    def call(self, x, adj=None):
        shape = tf.shape(x)
        B, N = shape[0], shape[1]
        if adj is None:
            h = tf.reduce_mean(x, axis=1, keepdims=True)  # shape (B, 1, F)
            h = tf.tile(h, [1, N, 1])  # broadcast to all nodes
        else:
            if isinstance(adj, tf.SparseTensor):
                h = tf.sparse.sparse_dense_matmul(adj, x)
            else:
                h = tf.matmul(adj, x)

        return self.dense(h)

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_dim": self.out_dim,
        })
        return config