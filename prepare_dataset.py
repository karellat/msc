import logging
import tensorflow as tf
import numpy as np
import os
import math

logger = logging.getLogger()
logger.setLevel(logging.INFO)

CLASS_NAMES = np.array(['ad', 'mci', 'cn'])
AUTOTUNE = tf.data.experimental.AUTOTUNE


BATCH_SIZE=128
DATASET_SIZE=21140
DEFAULT_PATH='/ADNI/slice_minc/*/*/*/*.png'

def get_label(file_path, class_folder=3): 
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[class_folder] == CLASS_NAMES

def decode_img(img, out_shape=(193,193)): 
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    if out_shape is None: 
        return img
    else:
        # Resize
        return tf.image.resize(img, out_shape)

def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    
    return img, label


def prepare_for_training(ds,
                         dataset_size,
                         cache=True,
                         shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
        ds = ds.cache(cache)
    else:
        ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    valid_size = int(math.floor(dataset_size/10.0))
    train_ds = ds.skip(valid_size*2)
    valid_ds = ds.take(valid_size*2).skip(valid_size)
    test_ds  = ds.take(valid_size)


    train_ds = train_ds.batch(BATCH_SIZE)
    valid_ds = valid_ds.batch(BATCH_SIZE)
    test_ds  = test_ds.batch(BATCH_SIZE)
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    valid_ds = valid_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    

    return train_ds, valid_ds, test_ds

def get_datasets(path=DEFAULT_PATH, dataset_size=DATASET_SIZE):
    list_ds = tf.data.Dataset.list_files(path)
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    return prepare_for_training(labeled_ds, dataset_size=dataset_size)
