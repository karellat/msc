# Training
import tensorflow as tf
from tensorflow.keras import layers

def get_baseline(
        filters=[16, 32],
        num_features=2,
        fc_num=16,
        stride=(2,2,2),
        ):
    kernel = 3
    activation = 'relu'
    pool_kernel = (3, 3, 3)
    dropout = 0.5
    padding='same'

    img_inputs = layers.Input((256, 256, 166, 1))
    last_layer = img_inputs
    for fc in filters:
        conv = layers.Conv3D(fc,
                             kernel,
                             strides=stride,
                             padding=padding,
                             activation=activation)(last_layer)
        maxp = layers.MaxPool3D()(conv)
        last_layer = maxp
    flatten = layers.Flatten()(last_layer)

    fc  = layers.Dense(fc_num, activation='relu')(flatten)
    dr_fc = layers.Dropout(dropout)(fc)

    output = layers.Dense(num_features, activation='softmax')(dr_fc)

    return(tf.keras.Model(
        inputs=img_inputs,
        outputs=output,
        name='3D_Dense'))
