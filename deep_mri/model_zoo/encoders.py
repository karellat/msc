import tensorflow as tf
import numpy as np


def autoencoder_conv_model(input_shape=(5, 5, 5, 1),
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


def conv_3d_baseline(conv_kernel=(5, 5, 5),
                     conv_filters=[16, 32, 64, 128],
                     pool_kernel=(5, 5, 5),
                     pool_stride=(3, 3, 3),
                     padding='same',
                     conv_activation='relu'):
    return tf.keras.Sequential([
        tf.keras.layers.Input((192, 223, 192, 1)),
        tf.keras.layers.Convolution3D(conv_filters[0], conv_kernel, padding=padding, activation=conv_activation),
        tf.keras.layers.MaxPool3D(pool_size=pool_kernel, strides=pool_stride, padding=padding),
        tf.keras.layers.Convolution3D(conv_filters[1], conv_kernel, padding=padding, activation=conv_activation),
        tf.keras.layers.MaxPool3D(pool_size=pool_kernel, strides=pool_stride, padding=padding),
        tf.keras.layers.Convolution3D(conv_filters[2], conv_kernel, padding=padding, activation=conv_activation),
        tf.keras.layers.MaxPool3D(pool_size=pool_kernel, strides=pool_stride, padding=padding),
        tf.keras.layers.Convolution3D(conv_filters[3], conv_kernel, padding=padding, activation=conv_activation),
        tf.keras.layers.MaxPool3D(pool_size=pool_kernel, strides=pool_stride, padding=padding),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
