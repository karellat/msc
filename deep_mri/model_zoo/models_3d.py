import tensorflow as tf


def batch_norm_conv(input, filters, kernel, name, activation):
    layer = tf.keras.layers.Convolution3D(filters, kernel, name=name, activation=None)(input)
    layer = tf.keras.layers.BatchNormalization()(layer)
    return tf.keras.layers.Activation(activation)(layer)


def payan_montana_model(input_shape=(97, 115, 97, 1),
                        conv_filters_count=150,
                        batch_norm=True,
                        dropout=0.5,
                        fc_size=800):
    # TODO: add paper name
    assert len(input_shape) == 4

    input_layer = tf.keras.layers.Input(input_shape, name=f'Input')
    if batch_norm:
        conv_layer1 = batch_norm_conv(input_layer, conv_filters_count, (5, 5, 5), name='Conv-1', activation='relu')
        conv_layer2 = batch_norm_conv(input_layer, conv_filters_count, (5, 5, 5), name='Conv-2', activation='relu')
        conv_layer3 = batch_norm_conv(input_layer, conv_filters_count, (5, 5, 5), name='Conv-3', activation='relu')
    else:
        conv_layer1 = tf.keras.layers.Convolution3D(conv_filters_count, (5, 5, 5), name='Conv-1', activation='relu')(
            input_layer)
        conv_layer2 = tf.keras.layers.Convolution3D(conv_filters_count, (5, 5, 5), name='Conv-2', activation='relu')(
            input_layer)
        conv_layer3 = tf.keras.layers.Convolution3D(conv_filters_count, (5, 5, 5), name='Conv-3', activation='relu')(
            input_layer)
    maxp_layer1 = tf.keras.layers.MaxPool3D(name='MaxP-1', pool_size=(5, 5, 5))(conv_layer1)
    maxp_layer2 = tf.keras.layers.MaxPool3D(name='MaxP-2', pool_size=(5, 5, 5))(conv_layer2)
    maxp_layer3 = tf.keras.layers.MaxPool3D(name='MaxP-3', pool_size=(5, 5, 5))(conv_layer3)
    maxp_layer1 = tf.keras.layers.Flatten()(maxp_layer1)
    maxp_layer2 = tf.keras.layers.Flatten()(maxp_layer2)
    maxp_layer3 = tf.keras.layers.Flatten()(maxp_layer3)
    conv_layer = tf.keras.layers.Concatenate(name='Concat')([maxp_layer1, maxp_layer2, maxp_layer3])
    flat_layer = tf.keras.layers.Flatten()(conv_layer)
    fc_layer = tf.keras.layers.Dense(fc_size, name=f'FC-{fc_size}', activation='relu')(flat_layer)
    drop_layer = tf.keras.layers.Dropout(dropout)(fc_layer)
    output_layer = tf.keras.layers.Dense(3, name='Classification', activation='softmax')(drop_layer)

    return tf.keras.Model(input_layer, output_layer)


def payan_montana_model_pretrained_conv(path_to_model, input_shape=(97, 115, 97, 1), fc_size=800):
    pretrained_model = tf.keras.models.load_model(path_to_model)
    trained_layer = pretrained_model.get_layer('Encoder-Conv')
    big_model = payan_montana_model(input_shape=input_shape,
                                    fc_size=fc_size,
                                    conv_filters_count=trained_layer.get_config()['filters'])
    pretrained_weights = trained_layer.get_weights()
    big_model.get_layer('Conv-1').trainable = False
    big_model.get_layer('Conv-2').trainable = False
    big_model.get_layer('Conv-3').trainable = False
    big_model.get_layer('Conv-1').set_weights(pretrained_weights)
    big_model.get_layer('Conv-2').set_weights(pretrained_weights)
    big_model.get_layer('Conv-3').set_weights(pretrained_weights)

    return big_model



