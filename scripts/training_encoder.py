from deep_mri.model_zoo.encoders import autoencoder_conv_model
from deep_mri.dataset.dataset import DEFAULT_PATH
import tensorflow as tf
import os
from deep_mri.dataset.dataset_encoder import get_encoder_dataset, DEFAULT_GENERATOR_ARGS
from datetime import datetime


def train(batch_size,
          epochs,
          conv_filters,
          regularization,
          conv_activation,
          dense_activation,
          input_shape=(5, 5, 5, 1),
          conv_kernel=(5, 5, 5),
          boxes_per_img=100,
          downscale_ratio=None,
          model_name=None,
          log_root=os.path.join("encoder", "logs")):
    model_name = model_name if model_name is not None else datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_dir = os.path.join(log_root, model_name)
    models_dir = os.path.join(log_dir, "models")

    dataset_generator_args = {
        "normalize": True,
        "box_size": input_shape[0],
        "downscale_ratio": downscale_ratio,
        "boxes_per_img": boxes_per_img,
    }
    train_ds, valid_ds = get_encoder_dataset(path=DEFAULT_PATH, **dataset_generator_args)

    model = autoencoder_conv_model(input_shape=input_shape,
                                   conv_kernel=conv_kernel,
                                   conv_filters=conv_filters,
                                   regularization=regularization,
                                   conv_activation=conv_activation,
                                   dense_activation=dense_activation)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    model_save_callback = tf.keras.callbacks.ModelCheckpoint(models_dir, save_best_only=True)
    earlystopping_callback = tf.keras.callbacks.EarlyStopping(patience=10)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['mse'])

    model.fit(train_ds.batch(batch_size),
              epochs=epochs,
              validation_data=valid_ds.batch(batch_size),
              callbacks=[tensorboard_callback, model_save_callback, earlystopping_callback],
              workers=40,
              use_multiprocessing=True,
              )
