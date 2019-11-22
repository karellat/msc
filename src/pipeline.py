import numpy as np
import tensorflow as tf

from logging import info, warning, error
from config import *
from reader import nii_dir_generator
from normalize import normalize
from model import get_baseline

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
    labels.append(fname)
    images.append(img)

images = np.array(images)
labels = np.array(labels)

info('Reading finished')

# NORMALIZATION PHASE
voxel_mean = np.mean(images)
voxel_std = np.std(images)
voxel_max = np.max(images)
voxel_min = np.min(images)

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
labels = labels == 'CN'

train_x = images[2:]
train_y = labels[2:]

test_x = images[:2]
test_y = labels[:2]

val_x = test_x
val_y = test_y

info('Preparation finished')

# TRAINING PHASE
# TODO: USE Straka logging name trick
callbacks = [tf.keras.callbacks.TensorBoard(log_dir=T_LOGS),
             tf.keras.callbacks.ModelCheckpoint(filepath=T_CHECKPOINT,
                                                verbose=1)
             ]
model = get_baseline()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.optimizers.Adam(),
              metrics=['accuracy'])
history = model.fit(train_x, train_y,
                    batch_size=T_BATCH_SIZE,
                    epochs=T_EPOCHS,
                    validation_data=(val_x, val_y),
                    callbacks=callbacks)
# EVALUATE PHASE
print(f'Test')
test_scores = model.evaluate(test_x, test_y, batch_size=T_BATCH_SIZE)
print(f'Test loss: {test_scores[0]}')
print(f'Test accuracy: {test_scores[1]}')