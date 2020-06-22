import tensorflow as tf
from deep_mri.model_zoo.resnet3d import Resnet3DBuilder


def batch_norm_conv(input, filters, kernel, name, activation):
    layer = tf.keras.layers.Convolution3D(filters, kernel, name=name, activation=None)(input)
    layer = tf.keras.layers.BatchNormalization()(layer)
    return tf.keras.layers.Activation(activation)(layer)


def payan_montana_model(input_shape=(97, 115, 97, 1),
                        conv_filters_count=150,
                        batch_norm=False,
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


def encoder_baseline(input_shape=(97, 115, 97, 1),
                     init_filters=150,
                     fc_size=500,
                     dropout=0.5):
    input_layer = tf.keras.layers.Input(input_shape)
    conv1_layer = tf.keras.layers.Conv3D(filters=init_filters,
                                         kernel_size=3,
                                         strides=1,
                                         activation='elu', )(input_layer)
    conv1_layer = tf.keras.layers.MaxPool3D(3)(conv1_layer)
    conv2_layer = tf.keras.layers.Conv3D(filters=init_filters,
                                         kernel_size=3,
                                         strides=1,
                                         activation='tanh', )(input_layer)
    conv2_layer = tf.keras.layers.MaxPool3D(3)(conv2_layer)
    mul_layer = tf.keras.layers.Multiply()([conv1_layer, conv2_layer])
    flat_layer = tf.keras.layers.Flatten()(mul_layer)
    fc_layer = tf.keras.layers.Dense(fc_size, name=f'FC-{fc_size}', activation='relu')(flat_layer)
    drop_layer = tf.keras.layers.Dropout(dropout)(fc_layer)
    output_layer = tf.keras.layers.Dense(3, name='Classification', activation='softmax')(drop_layer)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer)


def encoder_fc(encoder_model_path,
               pretrained_layers=['conv1', 'maxp1', 'conv2', 'maxp2', 'conv3', 'maxp3'],
               input_shape=(93, 115, 93, 1),
               fc_units=800):
    pretrained_model = tf.keras.models.load_model(encoder_model_path)
    encoder_layer = []
    for n in pretrained_layers:
        layer = pretrained_model.get_layer(n)
        if n.startswith('conv'):
            encoder_layer.append(tf.keras.layers.Conv3D.from_config(layer.get_config()))
        elif n.startswith('maxp'):
            encoder_layer.append(tf.keras.layers.MaxPool3D.from_config(layer.get_config()))

    encoder_layer.append(tf.keras.layers.Flatten())
    encoder_layer.append(tf.keras.layers.Dense(fc_units, activation='relu'))
    encoder_layer.append(tf.keras.layers.Dense(3, activation='softmax'))

    model = tf.keras.Sequential([tf.keras.layers.Input(shape=input_shape)] + encoder_layer)

    for n in pretrained_layers:
        w = pretrained_model.get_layer(n).get_weights()
        layer = model.get_layer(n)
        layer.set_weights(w)
        layer.trainable = False

    return model


def martin_model(input_shape=(97, 115, 97, 1),
                 init_filters=256,
                 conv_layers=5,
                 conv_kernel=2,
                 conv_activation='relu',
                 conv_stride=2,
                 conv_to_fc=None):
    convs = []
    for i in range(conv_layers):
        convs.append(tf.keras.layers.Conv3D(init_filters * (2 ** i),
                                            conv_kernel,
                                            conv_stride,
                                            activation=conv_activation,
                                            padding='same'))

    layers = [
        tf.keras.layers.Input(input_shape),
        *convs,
    ]
    if conv_to_fc == 'avg':
        layers.append(tf.keras.layers.AveragePooling3D())
    elif conv_to_fc == 'max':
        layers.append(tf.keras.layers.MaxPool3D())
    elif conv_to_fc == None:
        pass
    else:
        raise NotImplementedError(f'Unknown conv to fc transition {conv_to_fc}')
    model = tf.keras.Sequential(layers + [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(800, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    return model


def factory(model_name, **model_args):
    if model_name.lower() == "martin":
        return martin_model(**model_args)
    elif model_name.lower() == "payan":
        return payan_montana_model(**model_args)
    elif model_name.lower() == "baseline":
        return encoder_baseline(**model_args)
    elif model_name.lower() == "encoderfc":
        return encoder_fc(**model_args)
    elif model_name.lower() == "pretrained_payan":
        return payan_montana_model_pretrained_conv(**model_args)
    elif model_name.lower() == "resnet":
        return Resnet3DBuilder.build(**model_args)
    elif model_name.lower() == "resnet18":
        return Resnet3DBuilder.build_resnet_18(**model_args)
    elif model_name.lower() == "resnet34":
        return Resnet3DBuilder.build_resnet_34(**model_args)
    elif model_name.lower() == "resnet50":
        return Resnet3DBuilder.build_resnet_50(**model_args)
    elif model_name.lower() == "resnet101":
        return Resnet3DBuilder.build_resnet_101(**model_args)
    elif model_name.lower() == "resnet152":
        return Resnet3DBuilder.build_resnet_152(**model_args)
    else:
        raise Exception(f"Unknown 3d model : {model_name}")
