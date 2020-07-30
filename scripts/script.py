from deep_mri.dataset.dataset_3d import get_3d_dataset
from deep_mri.dataset.dataset import DEFAULT_PATH, get_all_files
from deep_mri.model_zoo.models_3d import payan_montana_model
import re
import glob
import numpy as np
import logging
import tensorflow as tf

logging.basicConfig(level=logging.WARN)

BATCH_SIZE = 2
EPOCHS=100
DOWNSCALE_RATIO=3.0
DROPOUT=0.5
BATCH_NORM=True
FC = 64 

IMAGE_SHAPE=np.ceil(np.array((193, 229, 193,1))/DOWNSCALE_RATIO).astype(int)
IMAGE_SHAPE=tuple(IMAGE_SHAPE)

LOG_DIR = f"3d_models/logs/fc{FC}-d{DROPOUT}-bn{BATCH_NORM}"
MODELS_DIR = LOG_DIR + "/models"
file_writer_cm = tf.summary.create_file_writer(LOG_DIR + '/cm')

file_writer = tf.summary.create_file_writer(LOG_DIR + "/metrics")
file_writer.set_as_default()

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)

train_ds, valid_ds, test_ds = get_3d_dataset(get_all_files(filter_first_screen=True),
                                            downscale_ratio=DOWNSCALE_RATIO)

model = payan_montana_model(input_shape=IMAGE_SHAPE, fc_size=FC, dropout=DROPOUT, batch_norm=BATCH_NORM)
model.summary()
model.compile(optimizer='adam',
               loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalAccuracy()])

model.fit(train_ds.batch(BATCH_SIZE),
                    epochs=EPOCHS,
                    validation_data=valid_ds.batch(BATCH_SIZE),
                    verbose=1,
                    workers=40,
                    use_multiprocessing=True,
                    callbacks=[tensorboard_callback]
                   )
