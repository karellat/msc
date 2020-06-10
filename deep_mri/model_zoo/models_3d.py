import tensorflow as tf

def payan_montana_model(input_shape=(97, 115, 97, 1),
                        conv_filters_count=150,
                        fc_size=800):
    #TODO: add paper name
    assert len(input_shape) == 4

    input_layer = tf.keras.layers.Input(input_shape, name=f'Input')
    conv_layer1 = tf.keras.layers.Convolution3D(conv_filters_count, (5,5,5), name='Conv-1', activation='relu')(input_layer)
    conv_layer2 = tf.keras.layers.Convolution3D(conv_filters_count, (5,5,5), name='Conv-2', activation='relu')(input_layer)
    conv_layer3 = tf.keras.layers.Convolution3D(conv_filters_count, (5,5,5), name='Conv-3', activation='relu')(input_layer)
    maxp_layer1 = tf.keras.layers.MaxPool3D(name='MaxP-1', pool_size=(5,5,5))(conv_layer1)
    maxp_layer2 = tf.keras.layers.MaxPool3D(name='MaxP-2', pool_size=(5,5,5))(conv_layer2)
    maxp_layer3 = tf.keras.layers.MaxPool3D(name='MaxP-3', pool_size=(5,5,5))(conv_layer3)
    maxp_layer1 = tf.keras.layers.Flatten()(maxp_layer1)
    maxp_layer2 = tf.keras.layers.Flatten()(maxp_layer2)
    maxp_layer3 = tf.keras.layers.Flatten()(maxp_layer3)
    conv_layer = tf.keras.layers.Concatenate(name='Concat')([maxp_layer1, maxp_layer2, maxp_layer3])
    flat_layer = tf.keras.layers.Flatten()(conv_layer)
    fc_layer = tf.keras.layers.Dense(fc_size,name=f'FC-{fc_size}', activation='relu')(flat_layer)
    output_layer = tf.keras.layers.Dense(3, name='Classification', activation='softmax')(fc_layer)

    return tf.keras.Model(input_layer, output_layer)