# Training
import tensorflow as tf
from tensorflow.keras import layers

def get_baseline():
    img_inputs = layers.Input((256, 256, 166, 1))
    conv0 = layers.Conv3D(16,
                          3,
                          strides=(2,2,2),
                          activation='relu')(img_inputs)
    conv1 = layers.Conv3D(32,
                          3,
                          strides=(2,2,2),
                          activation='relu')(conv0)
    conv2 = layers.Conv3D(64,
                          3,
                          strides=(2,2,2),
                          activation='relu')(conv1)
    conv3 = layers.Conv3D(128,
                          3,
                          strides=(2,2,2),
                          activation='relu')(conv2)
    conv4 = layers.Conv3D(256,
                          3,
                          strides=(2,2,2),
                          activation='relu')(conv3)
    flatten = layers.Flatten()(conv4)

    output = layers.Dense(2, activation='softmax')(flatten)

    return(tf.keras.Model(inputs=img_inputs, outputs=output, name='3D_Dense'))