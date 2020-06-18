import tensorflow as tf
import numpy as np
from tensorflow.math import log


def cae(input_shape=(32, 32, 32, 1), init_filters=256, activation='relu'):
    return tf.keras.Sequential([
        tf.keras.layers.Input(input_shape),
        tf.keras.layers.Conv3D(filters=init_filters, kernel_size=6, strides=2, activation=activation),
        tf.keras.layers.Conv3D(filters=init_filters * 2, kernel_size=6, strides=2, activation=activation),
        tf.keras.layers.Conv3D(filters=init_filters * 4, kernel_size=5, activation=activation),
        tf.keras.layers.Conv3DTranspose(filters=init_filters * 2, kernel_size=5, activation=activation),
        tf.keras.layers.Conv3DTranspose(filters=init_filters, kernel_size=6, strides=2, activation=activation),
        tf.keras.layers.Conv3DTranspose(filters=1, kernel_size=6, strides=2, activation=activation)
    ])


def autoencoder_baseline(input_shape=(5, 5, 5, 1),
                         conv_kernel=(5, 5, 5),
                         conv_activation='relu',
                         regularization='l2',
                         conv_filters=8,
                         dense_activation=None):
    fc_nodes = np.prod(input_shape)
    return tf.keras.Sequential([
        tf.keras.layers.Input(input_shape, name='Input'),
        tf.keras.layers.Conv3D(conv_filters, conv_kernel, name='Encoder-Conv', activation=conv_activation,
                               kernel_regularizer=regularization),
        tf.keras.layers.Dense(fc_nodes, activation=dense_activation, name='Decoder'),
        tf.keras.layers.Reshape(input_shape, name='output')
    ])


def _kld(target_s, s):
    return target_s * log(target_s / s) + (1 - target_s) * log((1 - target_s) / (1 - s))


class PayanEncoder(tf.keras.Model):
    def __init__(self,
                 input_shape=(5, 5, 5, 1),
                 conv_filters=150,
                 conv_kernel=(5, 5, 5),
                 sparsity=0.5,
                 conv_activation='sigmoid',
                 name="PayanModel",
                 **kwargs):
        fc_nodes = np.product(input_shape)
        super(PayanEncoder, self).__init__(name=name, **kwargs)
        self.conv = tf.keras.layers.Conv3D(filters=conv_filters,
                                           kernel_size=conv_kernel,
                                           activation=conv_activation,
                                           input_shape=input_shape,
                                           name='Encoder-Conv',
                                           kernel_regularizer='l2')
        self.fc = tf.keras.layers.Dense(units=fc_nodes,
                                        activation=None,
                                        name='Decoder-Dense')
        self.out = tf.keras.layers.Reshape(target_shape=input_shape,
                                           name='Out')
        # Loss
        self.mse = tf.keras.losses.MeanSquaredError()
        self.kl = tf.keras.losses.KLDivergence()
        self.sparsity = sparsity
        self.alpha = 0.5
        self.beta = 0.5

    def call(self, inputs, training=None, mask=None):
        encoder_l = self.conv(inputs)
        decoder_l = self.fc(encoder_l)
        out_l = self.out(decoder_l)
        activation_mean = tf.reduce_mean(encoder_l, axis=0)
        elems = (tf.fill(value=self.sparsity, dims=activation_mean.shape),
                 activation_mean)
        kl_divergence = tf.map_fn(lambda x: _kld(x[0], x[1]),
                                  elems,
                                  dtype=tf.float32)
        kl = tf.reduce_sum(kl_divergence)
        self.add_loss(self.beta * kl)

        return out_l


def my_encoder(init_filters=256,
               input_shape=(32, 32, 32, 1)):
    pretrained_layers = ['conv1', 'maxp1', 'conv2', 'maxp2', 'conv3', 'maxp3']
    return tf.keras.Sequential([
        tf.keras.layers.Input(input_shape),
        tf.keras.layers.Conv3D(filters=init_filters,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               name=pretrained_layers[0]),
        tf.keras.layers.MaxPool3D(3, name=pretrained_layers[1]),
        tf.keras.layers.Conv3D(filters=init_filters * 2,
                               kernel_size=3,
                               strides=1,
                               activation='relu',
                               name=pretrained_layers[2]),
        tf.keras.layers.MaxPool3D(2, name=pretrained_layers[3]),
        tf.keras.layers.Conv3D(filters=init_filters * 4,
                               kernel_size=3,
                               activation='relu',
                               name=pretrained_layers[4]),
        tf.keras.layers.MaxPool3D(2, name=pretrained_layers[5]),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(np.product(input_shape), activation='relu'),
        tf.keras.layers.Reshape(input_shape)
    ])


def factory(model_name, **model_args):
    if model_name.lower() == 'baseline':
        return autoencoder_baseline(**model_args)
    elif model_name.lower() == 'myencoder':
        return my_encoder(**model_args)
    elif model_name.lower() == 'payan':
        return PayanEncoder(**model_args)
    elif model_name.lower() == "cae":
        return cae(**model_args)
    else:
        raise Exception(f"Unknown encoder name {model_name}")
