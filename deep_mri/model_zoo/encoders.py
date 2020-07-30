import tensorflow as tf
import numpy as np
from tensorflow.math import log


def cae(input_shape=(32, 32, 32, 1), init_filters=256, activation='relu'):
    """
    Convolutional encoder using Transposed Convolutions as decoder

    Parameters
    ----------
    input_shape : tuple
        Expected input shape
    init_filters : int
        Filters of the first convolutional layer
    activation : string
        Tensorflow activation function for all the layers

    Returns
    -------
    tf.keras.model

    """
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
    """
    Convolutional encoder using Dense layer as decoder

    Parameters
    ----------
    input_shape : tuple
        Expected input shape
    conv_kernel : tuple
        Shape of the convolutional kernel
    conv_activation : string
        Tensorflow activation function for all the layers
    regularization : string
        Regularization of convolutional weights
    conv_filters : int
        Number of convolutional filters
    dense_activation : string
        Decoder activation function

    Returns
    -------
    tf.keras.model
    """
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
    """
    Payan Encoder model
    """
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


def payan2():
    """
    Payan model with default settings

    Returns
    -------
    tf.keras.model
    """
    input_layer = tf.keras.layers.Input((5, 5, 5, 1))
    conv1 = tf.keras.layers.Conv3D(150, 5, activation='sigmoid', padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.1))(input_layer)
    conv2 = tf.keras.layers.Conv3D(150, 5, activation='sigmoid', padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.1))(input_layer)
    conv3 = tf.keras.layers.Conv3D(150, 5, activation='sigmoid', padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.1))(input_layer)

    max1 = tf.keras.layers.MaxPool3D(5)(conv1)
    max2 = tf.keras.layers.MaxPool3D(5)(conv2)
    max3 = tf.keras.layers.MaxPool3D(5)(conv3)
    conc = tf.keras.layers.Concatenate()([max1, max2, max3])
    flat = tf.keras.layers.Flatten()(conc)
    fc = tf.keras.layers.Dense(125, activation='sigmoid')(flat)
    rs = tf.keras.layers.Reshape((5, 5, 5, 1))(fc)

    return tf.keras.Model(inputs=input_layer, outputs=rs)


def my_encoder(init_filters=256,
               input_shape=(32, 32, 32, 1)):
    """
    CNN encoder model
    Parameters
    ----------
    init_filters : int
        Number of filters
    input_shape : tuple
        Expected input shape
    Returns
    -------
    tf.keras.model

    """
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
    """
    Encoder factory creating encoder by given name and passing the model args

    Parameters
    ----------
    model_name : str
        Name of the encoder model
    model_args : dict
        Arguments for model

    Returns
    -------
    tf.keras.model

    """
    if model_name.lower() == 'baseline':
        return autoencoder_baseline(**model_args)
    elif model_name.lower() == 'myencoder':
        return my_encoder(**model_args)
    elif model_name.lower() == 'payan':
        return PayanEncoder(**model_args)
    elif model_name.lower() == 'payan2':
        return payan2(**model_args)
    elif model_name.lower() == "cae":
        return cae(**model_args)
    else:
        raise Exception(f"Unknown encoder name {model_name}")
