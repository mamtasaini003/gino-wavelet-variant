import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

@keras.utils.register_keras_serializable()
class FourierNeuralOperator(layers.Layer):
    def __init__(self, out_dim, modes=32):
        super().__init__()
        self.out_dim = out_dim
        self.modes = modes
        self.dense = layers.Dense(out_dim)

    def build(self, input_shape):
        _, N, F = input_shape
        self.weight_real = self.add_weight(
            shape=(self.modes, F, self.out_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=True,
            name="weight_real"
        )
        self.weight_imag = self.add_weight(
            shape=(self.modes, F, self.out_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=True,
            name="weight_imag"
        )

    def call(self, x):
        x_complex = tf.cast(x, tf.complex64)
        x_fft = tf.signal.fft(tf.transpose(x_complex, [0,2,1]))
        x_fft /= tf.cast(x_fft.shape[-1], tf.complex64)

        x_fft_low = x_fft[:, :, :self.modes]
        real = tf.math.real(x_fft_low)
        imag = tf.math.imag(x_fft_low)

        out_real = tf.einsum('bfm,mfd->bfd', real, self.weight_real) - tf.einsum('bfm,mfd->bfd', imag, self.weight_imag)
        out_imag = tf.einsum('bfm,mfd->bfd', real, self.weight_imag) + tf.einsum('bfm,mfd->bfd', imag, self.weight_real)

        out_fft_low = tf.complex(out_real, out_imag)

        pad_len = x_fft.shape[-1] - self.modes
        out_fft = tf.concat([out_fft_low, tf.zeros((tf.shape(out_fft_low)[0], tf.shape(out_fft_low)[1], pad_len), dtype=tf.complex64)], axis=-1)

        out_ifft = tf.signal.ifft(out_fft)
        out_ifft = tf.math.real(out_ifft) * tf.cast(out_ifft.shape[-1], tf.float32)
        out_ifft = tf.transpose(out_ifft, [0,2,1])

        return self.dense(out_ifft)

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_dim": self.out_dim,
            "modes" : self.modes
        })
        return config
    