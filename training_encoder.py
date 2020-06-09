from dataset import DEFAULT_PATH, AUTOTUNE
import tensorflow as tf
import logging
import numpy as np
from prepare_encoder_dataset import get_encoder_dataset, DEFAULT_GENERATOR_ARGS

tf.config.list_physical_devices('GPU')

def autoencoder_conv_model(input_shape=(5,5,5,1),
                           conv_kernel=(5,5,5),
                           conv_activation='relu',
                           regularization='l2',
                           conv_filters=8,
                           dense_activation=None):
    fc_nodes = np.prod(input_shape)
    return tf.keras.Sequential([
    tf.keras.layers.Input(input_shape, name='Input'), 
    tf.keras.layers.Conv3D(conv_filters, (5,5,5),name='Encoder-Conv', activation=conv_activation, kernel_regularizer=regularization),
    tf.keras.layers.Dense(fc_nodes,activation=dense_activation, name='Decoder'),
    tf.keras.layers.Reshape(input_shape, name='output')
    ])


from itertools import product
conv_filters = [2, 8, 16, 32, 64 ,128]
regularization = ['l1', 'l2', 'l1_l2']
conv_activation = [None, 'relu', 'sigmoid']
dense_activation = [None, 'relu', 'sigmoid']


train_ds, valid_ds = get_encoder_dataset(path=DEFAULT_PATH ,**DEFAULT_GENERATOR_ARGS)

for f, r, c_a, d_a in product(conv_filters, regularization, conv_activation, dense_activation):
    model_name = f'f{f}-r{r}-c_a{c_a}-d_a{d_a}'
    model = autoencoder_conv_model(conv_filters=f, regularization=r, conv_activation=c_a,dense_activation=d_a)


    logging.basicConfig(level=logging.WARN)

    BATCH_SIZE = 64
    EPOCHS=50

    LOG_DIR = f"encoder/logs/{model_name}"
    MODELS_DIR = LOG_DIR + "/models"

    file_writer_cm = tf.summary.create_file_writer(LOG_DIR + '/cm')
    file_writer = tf.summary.create_file_writer(LOG_DIR + "/metrics")
    file_writer.set_as_default()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)
    model_save_callback = tf.keras.callbacks.ModelCheckpoint(MODELS_DIR, save_best_only=True)
    earlystopping_callback = tf.keras.callbacks.EarlyStopping(patience=10)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['mse'])

    model.fit(train_ds.batch(BATCH_SIZE),
                        epochs=EPOCHS,
                        validation_data=valid_ds.batch(BATCH_SIZE),
                        callbacks=[tensorboard_callback],
                        workers=40,
                        use_multiprocessing=True,
                       )
