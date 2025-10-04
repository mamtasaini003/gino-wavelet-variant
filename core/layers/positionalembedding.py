import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

@keras.utils.register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def call(self, x):
        pos = tf.expand_dims(x, -1)
        i = tf.range(self.embed_dim, dtype=tf.float32)
        angle_rates = 1.0 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(self.embed_dim, tf.float32))
        angle_rads = pos * angle_rates
        sines = tf.sin(angle_rads[..., 0::2])
        cosines = tf.cos(angle_rads[..., 1::2])
        pos_emb = tf.concat([sines, cosines], axis=-1)
        pos_emb = tf.reshape(pos_emb, (tf.shape(x)[0], tf.shape(x)[1], -1))
        return pos_emb

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim
        })
        return config