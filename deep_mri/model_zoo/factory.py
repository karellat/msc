from deep_mri.model_zoo import models_3d, encoders, models_2d

MODEL_TYPE_DELIMETER = "_"


def model_factory(model_name, **model_args):
    """
    Factory for creating tensorflow models

    Parameters
    ----------
    model_name : str
        Name of the model, <type name>, possible types [3d, encoder, 2d, str]
    model_args : dict
        Arguments for the model

    Returns
    -------
    tf.keras.model

    """
    assert len(model_name.split(MODEL_TYPE_DELIMETER)) == 2, "Model name can have only single delimeter"
    mod_type, mod_name = model_name.split(MODEL_TYPE_DELIMETER)
    if mod_type.lower() == "3d":
        return models_3d.factory(mod_name, **model_args)
    elif mod_type.lower() == "encoder":
        return encoders.factory(mod_name, **model_args)
    elif mod_type.lower() == "2d":
        return models_2d.factory(mod_name, **model_args)
    elif mod_type.lower() == 'str':
        assert 'layers_text' in model_args
        return string_to_model(mod_name, **model_args)
    else:
        raise Exception(f"Unknown model name: {model_name}")


import tensorflow as tf

param_split = '-'
layers_split = ';'


def _get_pool(pool_text, input_layer):
    params = pool_text.lower().split(param_split)
    assert len(params) == 4, f'{pool_text}'
    if params[0] == 'pmax3':
        return tf.keras.layers.MaxPool3D(pool_size=int(params[1]),
                                         strides=int(params[2]),
                                         padding=str(params[3]))(input_layer)
    elif params[0] == 'pmax2':
        return tf.keras.layers.MaxPool2D(pool_size=int(params[1]),
                                         strides=int(params[2]),
                                         padding=str(params[3]))(input_layer)
    elif params[0] == 'pavg3':
        return tf.keras.layers.AvgPool3D(pool_size=int(params[1]),
                                         strides=int(params[2]),
                                         padding=str(params[3]))(input_layer)
    elif params[0] == 'pavg2':
        return tf.keras.layers.AvgPool2D(pool_size=int(params[1]),
                                         strides=int(params[2]),
                                         padding=str(params[3]))(input_layer)
    else:
        raise Exception(f'Unknown string {pool_text}')


def _get_conv(conv_text, input_layer):
    params = conv_text.lower().split(param_split)
    act = str(params[5]) if str(params[5]) != 'none' else None
    assert len(params) == 6, f'{conv_text}'
    if params[0] == 'c3':
        return tf.keras.layers.Conv3D(filters=int(params[1]),
                                      kernel_size=int(params[2]),
                                      strides=int(params[3]),
                                      padding=str(params[4]),
                                      activation=act)(input_layer)
    elif params[0] == 'c2':
        return tf.keras.layers.Conv2D(filters=int(params[1]),
                                      kernel_size=int(params[2]),
                                      strides=int(params[3]),
                                      padding=str(params[4]),
                                      activation=act)(input_layer)
    elif params[0] == 'cb3':
        conv = tf.keras.layers.Conv3D(filters=int(params[1]),
                                      kernel_size=int(params[2]),
                                      strides=int(params[3]),
                                      padding=str(params[4]),
                                      activation=None)(input_layer)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Activation(act)(conv)
        return conv
    elif params[0] == 'cb2':
        conv = tf.keras.layers.Conv2D(filters=int(params[1]),
                                      kernel_size=int(params[2]),
                                      strides=int(params[3]),
                                      padding=str(params[4]),
                                      activation=None)(input_layer)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Activation(act)(conv)
        return conv
    else:
        raise Exception(f'Unknown string {conv_text}')


def _get_fc(fc_text, input_layer):
    params = fc_text.lower().split(param_split)
    assert len(params) == 3, f'{fc_text}'
    if params[0] == 'fc':
        act = str(params[2]) if str(params[2]) != 'none' else None
        return tf.keras.layers.Dense(units=int(params[1]),
                                     activation=act)(input_layer)
    else:
        raise Exception(f'Unknown string {fc_text}')


def _get_layer(layer_text, input_layer):
    if layer_text.lower().startswith('c'):
        return _get_conv(layer_text, input_layer)
    elif layer_text.lower().startswith('p'):
        return _get_pool(layer_text, input_layer)
    elif layer_text.lower().startswith('fc'):
        return _get_fc(layer_text, input_layer)
    elif layer_text.lower() == 'flat':
        return tf.keras.layers.Flatten()
    else:
        raise Exception(f'Unknown string {layer_text}')


def string_to_model(model_name, layers_text, input_shape, num_output=3):
    """
    Generate model by input string

    Parameters
    ----------
    model_name : str
        Model name
    layers_text : str
        String representing model
    input_shape : tuple
        Expected shape of input data
    num_output : int
        number of outputs

    Returns
    -------
    tf.keras.model
    """
    input_layer = tf.keras.layers.Input(input_shape)
    layers_text = layers_text.replace('\n', '')
    layers_text = layers_text.split(layers_split)
    layers = []
    last_layer = input_layer
    for layer_text in layers_text:
        last_layer = _get_layer(layer_text, last_layer)
        layers.append(last_layer)
    last_layer = tf.keras.layers.Flatten()(last_layer)
    output_layer = tf.keras.layers.Dense(num_output, activation='softmax')(last_layer)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer, name=model_name)
