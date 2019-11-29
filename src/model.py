# Training
import tensorflow as tf
from tensorflow.keras import layers

def get_baseline(num_features=3):
    stride = (1, 1, 1)
    kernel = 3
    activation = 'relu'
    pool_kernel = (3, 3, 3)
    filters = (8, 16, 32, 16, 8)
    fc_num = 32
    dropout = 0.5

    assert len(filters) == 5

    img_inputs = layers.Input((256, 256, 166, 1))

    conv0 = layers.Conv3D(filters[0],
                          kernel,
                          strides=stride,
                          activation=activation)(img_inputs)
    mp0 = layers.MaxPool3D()(conv0)

    conv1 = layers.Conv3D(filters[1],
                          kernel,
                          strides=stride,
                          activation=activation)(mp0)
    mp1 = layers.MaxPool3D()(conv1)

    conv2 = layers.Conv3D(filters[2],
                          kernel,
                          strides=stride,
                          activation=activation)(mp1)
    mp2 = layers.MaxPool3D()(conv2)

    conv3 = layers.Conv3D(filters[3],
                          kernel,
                          strides=stride,
                          activation=activation)(mp2)
    mp3 = layers.MaxPool3D()(conv3)

    conv4 = layers.Conv3D(filters[4],
                          kernel,
                          strides=stride,
                          activation=activation)(mp3)
    mp4 = layers.MaxPool3D()(conv4)

    flatten = layers.Flatten()(mp4)

    fc  = layers.Dense(fc_num, activation='relu')(flatten)
    dr_fc = layers.Dropout(dropout)(fc)

    output = layers.Dense(num_features, activation='softmax')(dr_fc)

    return(tf.keras.Model(
        inputs=img_inputs,
        outputs=output,
        name='3D_Dense'))
