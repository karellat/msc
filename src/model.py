# Training
import tensorflow as tf
from tensorflow.keras import layers

def get_baseline():
    stride = (1,1,1)
    kernel = 3
    activation = 'relu'
    pool_kernel = (3,3,3)

    img_inputs = layers.Input((256, 256, 166, 1))
    
    conv0 = layers.Conv3D(32,
                          kernel,
                          strides=stride,
                          activation=activation)(img_inputs)
    mp0 = layers.MaxPool3D()(conv0) 
    
    conv1 = layers.Conv3D(64,
                          kernel,
                          strides=stride,
                          activation=activation)(mp0)
    mp1 = layers.MaxPool3D()(conv1)
    
    conv2 = layers.Conv3D(128,
                          kernel,
                          strides=stride,
                          activation=activation)(mp1)
    mp2 = layers.MaxPool3D()(conv2)
    
    conv3 = layers.Conv3D(64,
                          kernel,
                          strides=stride,
                          activation=activation)(mp2)
    mp3 = layers.MaxPool3D()(conv3)
    
    conv4 = layers.Conv3D(32,
                          kernel,
                          strides=stride,
                          activation=activation)(mp3)
    mp4 = layers.MaxPool3D()(conv4)

    flatten = layers.Flatten()(mp4)

    fc  = layers.Dense(1024, activation='relu')(flatten)
    dr_fc = layers.Dropout(0.8)(fc)

    output = layers.Dense(3, activation='softmax')(dr_fc)

    return(tf.keras.Model(inputs=img_inputs, outputs=output, name='3D_Dense'))
