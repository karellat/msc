# Training
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv3D, MaxPool3D, Dropout, Dense, BatchNormalization, Flatten
from tensorflow.keras import Sequential

def get_baseline(
        filters=[32, 64],
        num_features=2,
        fc_num=128,
        stride=(2,2,2),
        batch_norm=True,
        ):
    kernel = 3
    activation = 'relu'
    pool_kernel = (3, 3, 3)
    dropout = 0.5
    padding='same'
    relu_ind = 0

    img_inputs = layers.Input((110, 110, 110, 1))
    last_layer = img_inputs
    for fc in filters:
        conv = layers.Conv3D(fc,
                             kernel,
                             strides=stride,
                             padding=padding,
                             activation=None)(last_layer)
        if batch_norm:
            conv = layers.BatchNormalization()(conv)
        conv = layers.Activation(activation, name=activation+str(relu_ind))(conv)
        relu_ind += 1
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

def get_VoxCNN():
    """ Return the Keras model of the network
    """
    model = Sequential()
    # 1st Volumetric Convolutional block
    model.add(Conv3D(8, (3, 3, 3), activation='relu', padding='same', input_shape=(110, 110, 110, 1)))
    model.add(Conv3D(8, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    # 2nd Volumetric Convolutional block
    model.add(Conv3D(16, (3, 3, 3), activation='relu', padding='same'))
    model.add(Conv3D(16, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    # 3rd Volumetric Convolutional block
    model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
    model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
    model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    # 4th Volumetric Convolutional block
    model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    # 1th Deconvolutional layer with batchnorm and dropout for regularization
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))
    # 2th Deconvolutional layer
    model.add(Dense(64, activation='relu'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.7))
    # Output with softmax nonlinearity for classification
    model.add(Dense(2, activation='softmax'))
    return model