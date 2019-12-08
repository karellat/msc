# Training
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv3D, MaxPool3D, Dropout, Dense, BatchNormalization, Flatten, Input, Activation
from tensorflow.keras import Sequential, Model


def get_baseline(
        filters=[32, 64],
        num_features=2,
        fc_num=128,
        stride=(2, 2, 2),
        batch_norm=True,
):
    kernel = 3
    activation = 'relu'
    pool_kernel = (3, 3, 3)
    dropout = 0.5
    padding = 'same'
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
        conv = layers.Activation(activation, name=activation + str(relu_ind))(conv)
        relu_ind += 1
        maxp = layers.MaxPool3D()(conv)
        last_layer = maxp
    flatten = layers.Flatten()(last_layer)

    fc = layers.Dense(fc_num, activation='relu')(flatten)
    dr_fc = layers.Dropout(dropout)(fc)

    output = layers.Dense(num_features, activation='softmax')(dr_fc)

    return (tf.keras.Model(
        inputs=img_inputs,
        outputs=output,
        name='3D_Dense'))


# Residual and plain convolutional neural networks for 3D brain MRI classification
# By https://ieeexplore.ieee.org/document/7950647
def get_voxcnn():
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
    # model.add(BatchNormalization())
    # model.add(Dropout(0.7))
    # Output with softmax nonlinearity for classification
    model.add(Dense(2, activation='softmax'))
    return model


# Residual and plain convolutional neural networks for 3D brain MRI classification
# By https://ieeexplore.ieee.org/document/7950647
def get_resnetmri():
    # Input
    input_layer = Input((110, 110, 110, 1))
    # No residual conv 1
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding="same", activation=None)(input_layer)
    x = BatchNormalization()(x)
    x = Activation(activation='relu', name='relu0')(x)

    x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding="same", activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu', name='relu1')(x)

    res1 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding="same", activation=None)(x)
    # Res 1 
    x = BatchNormalization()(res1)
    x = Activation(activation='relu', name='relu2')(x)

    x = Conv3D(filters=64, kernel_size=(3, 3, 3), padding="same", activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu', name='relu3')(x)

    x = Conv3D(filters=64, kernel_size=(3, 3, 3), padding="same", activation=None)(x)
    # residual connection 1
    res2 = x + res1

    # Res 3
    x = BatchNormalization()(res2)
    x = Activation(activation='relu', name='relu4')(x)

    x = Conv3D(filters=64, kernel_size=(3, 3, 3), padding="same", activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu', name='relu5')(x)

    x = Conv3D(filters=64, kernel_size=(3, 3, 3), padding="same", activation=None)(x)

    res3 = x + res2

    x = BatchNormalization()(res3)
    x = Activation(activation='relu', name='relu6')(x)
    res4 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding="same", activation=None)(x)
    # residual connection 4
    x = BatchNormalization()(res4)
    x = Activation(activation='relu', name='relu7')(x)

    x = Conv3D(filters=64, kernel_size=(3, 3, 3), padding="same", activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu', name='relu8')(x)

    x = Conv3D(filters=64, kernel_size=(3, 3, 3), padding="same", activation=None)(x)

    # residual connection 5
    res5 = x + res4

    x = BatchNormalization()(res5)
    x = Activation(activation='relu', name='relu9')(x)

    x = Conv3D(filters=64, kernel_size=(3, 3, 3), padding="same", activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu', name='relu10')(x)

    x = Conv3D(filters=64, kernel_size=(3, 3, 3), padding="same", activation=None)(x)

    res6 = x + res5

    x = BatchNormalization()(res6)
    x = Activation(activation='relu', name='relu11')(x)

    res7 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding="same", activation=None)(x)

    x = BatchNormalization()(res7)
    x = Activation(activation='relu', name='relu12')(x)

    x = Conv3D(filters=128, kernel_size=(3, 3, 3), padding="same", activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu', name='relu13')(x)

    x = x + res7
    x = MaxPool3D((7, 7, 7))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', name='relu14')(x)
    output_layer = Dense(2, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output_layer, name="MRIResnet")


def get_model(name, *args, **kwargs):
    name = name.lower().strip()
    if name == "resnet":
        return get_resnetmri(*args, **kwargs)
    elif name == "voxcnn":
        return get_voxcnn(*args, **kwargs)
    elif name == "baseline":
        return get_baseline(*args, **kwargs)
    else:
        raise Exception(f'Unknown model name {name}')
