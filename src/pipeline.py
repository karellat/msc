import numpy as np
import tensorflow as tf
import logging
from logging import info, warning, error, basicConfig
import os
from datetime import datetime
from shutil import copyfile

from config import *
from reader import nii_dir_generator
from normalize import normalize
from model import get_baseline
import adni

# MODULES INFO 
basicConfig(level=logging.DEBUG)

info(f'Tensorflow {tf.__version__}')

logs_dir = os.path.join(T_LOGS, datetime.now().strftime("%y-%m-%H-%M"))
# READ PHASE
info(f'Reading from {IMG_PATH}')

labels = []
images = []
img_generator = nii_dir_generator(input_dir=IMG_PATH,
                                  fname2label=FNAME_TO_LABEL,
                                  image_ext=IMG_EXT,
                                  default_shape=IMG_SHAPE,
                                  ignore_shape=IMG_IGNORE_BAD_SHAPE)
for fname, img in img_generator:
    if img is None: continue
    if fname == 1: continue # Filter MCI 
    labels.append(fname)
    images.append(img)

images = np.array(images)
labels = np.array(labels)

assert len(images) > 0
info('Reading finished')

# LABELS STATS
unique, counts = np.unique(labels, return_counts=True)
max_perc = np.max(counts)/np.sum(counts)
info(f'Ration {max_perc} max class {adni.int_to_str(unique[np.argmax(counts)])}') 
for label, count in zip(unique, counts):
    info(f'Label {label} = {count}')

# NORMALIZATION PHASE
info('Calculating data boundaries')
voxel_mean = np.mean(images)
voxel_std = np.std(images)
voxel_max = np.max(images)
voxel_min = np.min(images)
info(f'Normalization by {NORM_METHOD}')

normalize(images,
          feature_range=(0, 1),
          method=NORM_METHOD,
          min_data=voxel_min,
          max_data=voxel_max,
          copy=False)

info('Normalization finished')

# DATA AUGMENTATION PHASE
# TODO: Implement

# CLASS BALANCING PHASE
# TODO: Implement

# PREPARATION PHASE

assert images.shape[-1] != 1

images = images.reshape((*images.shape, 1)).astype('float32')

info('Preparation finished')
info(f'\t X data shape {images.shape}')
info(f'\t y data shape {labels.shape}') 

# TRAINING PHASE
# TODO: USE Straka logging name trick
callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logs_dir),
             tf.keras.callbacks.ModelCheckpoint(filepath=T_CHECKPOINT,
                                                verbose=1)
             ]
model = get_baseline()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.optimizers.Adam(),
              metrics=['accuracy'])
info(f'Compile')
history = model.fit(images, labels,
                    batch_size=T_BATCH_SIZE,
                    epochs=T_EPOCHS,
                    validation_split=0.2,
                    callbacks=callbacks)
# EVALUATE PHASE
info(f'Test')
#test_scores = model.evaluate(test_x, test_y, batch_size=T_BATCH_SIZE)
#info(f'Test loss: {test_scores[0]}')
#info(f'Test accuracy: {test_scores[1]}')

info(f'Coping config to {logs_dir}')
copyfile('src/config.py', os.path.join(logs_dir,'config.py'))
