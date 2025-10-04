import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import os
from core.models.fno import FourierNeuralOperator 
from core.layers.positionalembedding import PositionalEmbedding
from core.models.gno import GraphNeuralOperator

# ---- GINO Model ----
@keras.utils.register_keras_serializable()
class GINO(keras.Model):
    def __init__(self, embed_dim=32, hidden_dim=128, num_fno_layers=10):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_fno_layers = num_fno_layers

        self.embed = PositionalEmbedding(embed_dim)
        self.input_mlp = layers.Dense(hidden_dim, activation="gelu")
        self.encoder_gno = GraphNeuralOperator(hidden_dim)

        # Stack multiple FNO layers with BatchNorm
        self.fno_layers = []
        for _ in range(num_fno_layers):
            self.fno_layers.append(FourierNeuralOperator(hidden_dim, modes=18))
            self.fno_layers.append(layers.BatchNormalization())

        self.decoder_gno = GraphNeuralOperator(hidden_dim)

        self.latent_mlp = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation="gelu"),
            layers.Dense(hidden_dim, activation="gelu")
        ])
        self.output_mlp = layers.Dense(1)

    def call(self, inputs, training=False):
        input_pts, input_vals, query_pts = inputs
        B, N, _ = input_pts.shape
        M = query_pts.shape[1]

        input_feat = self.embed(input_pts)
        query_feat = self.embed(query_pts)

        # ensure vals is (B, N, F)
        if len(input_vals.shape) == 4:
            input_vals = tf.squeeze(input_vals, axis=-1)

        input_feat = tf.concat([input_feat, input_vals], axis=-1)

        input_feat = self.input_mlp(input_feat)

        latent = self.encoder_gno(input_feat)

        for layer in self.fno_layers:
            if isinstance(layer, layers.BatchNormalization):
                latent = layer(latent, training=training)
            else:
                latent = layer(latent)

        latent = self.decoder_gno(latent)
        latent = tf.reduce_mean(latent, axis=1)

        latent_query = tf.repeat(tf.expand_dims(latent, 1), M, axis=1)
        combined = tf.concat([query_feat, latent_query], axis=-1)

        h = self.latent_mlp(combined)
        out = self.output_mlp(h)
        return out[..., 0]

    def get_config(self):
        return {
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_fno_layers": self.num_fno_layers,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def save_gino_model(self, path):
        self.save(path, save_format="tf")

    def load_gino_model(path):
        model = keras.models.load_model(path)
        return model

