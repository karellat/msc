import logging
from nibabel import Nifti2Image
from prepare_dataset_3D import get_3d_dataset
from model_zoo import payan_montana_model
import nibabel as nib
from nilearn.plotting import plot_anat
from nilearn.image import resample_img
import numpy as np
import tensorflow as tf
import glob
from train_utils import log_confusion_matrix


BATCH_SIZE = 32
EPOCHS=100
DOWNSCALE_RATION=3.0
IMAGE_SHAPE=np.ceil(np.array((193, 229, 193,1))/DOWNSCALE_RATION).astype(int)
IMAGE_SHAPE=tuple(IMAGE_SHAPE)
LOG_DIR = f"3d_models/logs/payan_montana-b{BATCH_SIZE}-e{EPOCHS}-s{DOWNSCALE_RATION}"
MODELS_DIR = LOG_DIR + "/models"
file_writer_cm = tf.summary.create_file_writer(LOG_DIR + '/cm')

file_writer = tf.summary.create_file_writer(LOG_DIR + "/metrics")
file_writer.set_as_default()

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)
model_save_callback = tf.keras.callbacks.ModelCheckpoint(MODELS_DIR, save_best_only=True)
cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

train_ds, valid_ds, test_ds = get_3d_dataset(downscale_ratio=DOWNSCALE_RATION)
model = payan_montana_model(input_shape=IMAGE_SHAPE)
model.compile(optimizer='adam',
               loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalAccuracy()])

history = model.fit(train_ds.batch(BATCH_SIZE),
                    epochs=EPOCHS,
                    validation_data=valid_ds.batch(BATCH_SIZE),
                    workers=40,
                    use_multiprocessing=True)