"""
Tensorflow 2 reimplementation of
https://github.com/JihongJu/keras-resnet3d
"""
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)
import six
from math import ceil
import tensorflow as tf


def _conv3d_relu(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", "l2")

    def f(input):
        return tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel_size,
                                      strides=strides, padding=padding,
                                      activation='relu',
                                      kernel_regularizer=kernel_regularizer)(input)

    return f


def _shortcut3d(input, residual):
    """3D shortcut to match input and residual and merges them with "sum"."""
    stride_dim1 = ceil(input.shape[DIM1_AXIS] / residual.shape[DIM1_AXIS])
    stride_dim2 = ceil(input.shape[DIM2_AXIS] / residual.shape[DIM2_AXIS])
    stride_dim3 = ceil(input.shape[DIM3_AXIS] / residual.shape[DIM3_AXIS])
    equal_channels = residual.shape[CHANNEL_AXIS] == input.shape[CHANNEL_AXIS]

    shortcut = input
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 \
            or not equal_channels:
        shortcut = tf.keras.layers.Conv3D(
            filters=residual.shape[CHANNEL_AXIS],
            kernel_size=(1, 1, 1),
            strides=(stride_dim1, stride_dim2, stride_dim3),
            kernel_regularizer='l2'
        )(input)
    return tf.keras.layers.add([shortcut, residual])


def _residual_block3d(block_function,
                      filters,
                      kernel_regularizer,
                      repetitions,
                      is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            strides = (1, 1, 1)
            if i == 0 and not is_first_layer:
                strides = (2, 2, 2)
            input = block_function(filters=filters, strides=strides,
                                   kernel_regularizer=kernel_regularizer,
                                   is_first_block_of_first_layer=(
                                           is_first_layer and i == 0)
                                   )(input)
        return input

    return f


def basic_block(filters,
                strides=(1, 1, 1),
                kernel_regularizer='l2',
                is_first_block_of_first_layer=False):
    """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl."""

    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = tf.keras.layers.Conv3D(filters=filters, kernel_size=(3, 3, 3),
                                           strides=strides, padding="same",
                                           kernel_regularizer=kernel_regularizer
                                           )(input)
        else:
            conv1 = _conv3d_relu(filters=filters,
                                 kernel_size=(3, 3, 3),
                                 strides=strides,
                                 kernel_regularizer=kernel_regularizer
                                 )(input)

        residual = _conv3d_relu(filters=filters, kernel_size=(3, 3, 3),
                                kernel_regularizer=kernel_regularizer
                                )(conv1)
        return _shortcut3d(input, residual)

    return f


def bottleneck(filters, strides=(1, 1, 1), kernel_regularizer='l2',
               is_first_block_of_first_layer=False):
    """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl."""

    def f(input):
        conv_1_1 = _conv3d_relu(filters=filters, kernel_size=(1, 1, 1),
                                strides=strides,
                                kernel_regularizer=kernel_regularizer
                                )(input)

        conv_3_3 = _conv3d_relu(filters=filters, kernel_size=(3, 3, 3),
                                kernel_regularizer=kernel_regularizer
                                )(conv_1_1)
        residual = _conv3d_relu(filters=filters * 4, kernel_size=(1, 1, 1),
                                kernel_regularizer=kernel_regularizer
                                )(conv_3_3)

        return _shortcut3d(input, residual)

    return f


def _handle_data_format():
    global DIM1_AXIS
    global DIM2_AXIS
    global DIM3_AXIS
    global CHANNEL_AXIS
    DIM1_AXIS = 1
    DIM2_AXIS = 2
    DIM3_AXIS = 3
    CHANNEL_AXIS = 4


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class Resnet3DBuilder(object):
    """ResNet3D."""

    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, first_filters=64):
        """Instantiate a vanilla ResNet3D keras model.

        # Arguments
            input_shape: Tuple of input shape in the format
            (conv_dim1, conv_dim2, conv_dim3, channels) if dim_ordering='tf'
            (filter, conv_dim1, conv_dim2, conv_dim3) if dim_ordering='th'
            num_outputs: The number of outputs at the final softmax layer
            block_fn: Unit block to use {'basic_block', 'bottlenack_block'}
            repetitions: Repetitions of unit blocks
        # Returns
            model: a 3D ResNet model that takes a 5D tensor (volumetric images
            in batch) as input and returns a 1D vector (prediction) as output.
        """
        _handle_data_format()
        if len(input_shape) != 4:
            raise ValueError("Input shape should be a tuple "
                             "(conv_dim1, conv_dim2, conv_dim3, channels) "
                             "for tensorflow as backend or "
                             "(channels, conv_dim1, conv_dim2, conv_dim3) "
                             "for theano as backend")

        block_fn = _get_block(block_fn)
        input = tf.keras.Input(shape=input_shape)
        # first conv
        conv1 = _conv3d_relu(filters=first_filters, kernel_size=(7, 7, 7),
                             strides=(2, 2, 2),
                             kernel_regularizer="l2"
                             )(input)
        pool1 = tf.keras.layers.MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2),
                                             padding="same")(conv1)

        # repeat blocks
        block = pool1
        filters = first_filters
        for i, r in enumerate(repetitions):
            block = _residual_block3d(block_fn, filters=filters,
                                      kernel_regularizer='l2',
                                      repetitions=r, is_first_layer=(i == 0)
                                      )(block)
            filters *= 2

        # average poll and classification
        pool2 = tf.keras.layers.AveragePooling3D(pool_size=(block.shape[DIM1_AXIS],
                                                            block.shape[DIM2_AXIS],
                                                            block.shape[DIM3_AXIS]),
                                                 strides=(1, 1, 1))(block)
        flatten1 = tf.keras.layers.Flatten()(pool2)
        if num_outputs > 1:
            dense = tf.keras.layers.Dense(units=num_outputs,
                                          activation="softmax",
                                          kernel_regularizer='l2')(flatten1)
        else:
            dense = tf.keras.layers.Dense(units=num_outputs,
                                          activation="sigmoid",
                                          kernel_regularizer='l2')(flatten1)

        model = tf.keras.Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs, filters=64):
        """Build resnet 18."""
        return Resnet3DBuilder.build(input_shape, num_outputs, basic_block,
                                     [2, 2, 2, 2], filters)

    @staticmethod
    def build_resnet_34(input_shape, num_outputs, filters=64):
        """Build resnet 34."""
        return Resnet3DBuilder.build(input_shape, num_outputs, basic_block,
                                     [3, 4, 6, 3], filters)

    @staticmethod
    def build_resnet_50(input_shape, num_outputs, filters=64):
        """Build resnet 50."""
        return Resnet3DBuilder.build(input_shape, num_outputs, bottleneck,
                                     [3, 4, 6, 3], filters)

    @staticmethod
    def build_resnet_101(input_shape, num_outputs, filters=64):
        """Build resnet 101."""
        return Resnet3DBuilder.build(input_shape, num_outputs, bottleneck,
                                     [3, 4, 23, 3], filters)

    @staticmethod
    def build_resnet_152(input_shape, num_outputs, filters=64):
        """Build resnet 152."""
        return Resnet3DBuilder.build(input_shape, num_outputs, bottleneck,
                                     [3, 8, 36, 3], filters)
