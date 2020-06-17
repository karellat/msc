import logging
import tensorflow as tf
import numpy as np
import os
import math
from enum import Enum, auto

logger = logging.getLogger()
logger.setLevel(logging.INFO)

CLASS_NAMES = np.array(['ad', 'mci', 'cn'])
AUTOTUNE = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 128
DATASET_SIZE = 21140

def _get_label(file_path, class_folder=3):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[class_folder] == CLASS_NAMES


class ImgReshape(Enum):
    RESIZE = auto()
    RESIZE_CROP_PAD = auto()
    RESIZE_PAD = auto()


def _decode_img(img,
                out_shape=(193, 193),
                reshape_method=ImgReshape.RESIZE):
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    if out_shape is None:
        return img
    else:
        if reshape_method == ImgReshape.RESIZE:
            return tf.image.resize(img, out_shape)
        elif reshape_method == ImgReshape.RESIZE_CROP_PAD:
            return tf.image.resize_with_crop_or_pad(img, target_height=out_shape[0], target_width=out_shape[1])
        elif reshape_method == ImgReshape.RESIZE_PAD:
            return tf.image.resize_with_pad(img, target_height=out_shape[0], target_width=out_shape[1])


def _process_path(file_path, out_shape=(193, 193), reshape_method=ImgReshape.RESIZE):
    label = _get_label(file_path)
    img = tf.io.read_file(file_path)
    img = _decode_img(img,
                      out_shape=out_shape,
                      reshape_method=reshape_method)

    return img, label


def _prepare_for_training(ds,
                          dataset_size,
                          shuffle=True,
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

        valid_size = int(math.floor(dataset_size / 10.0))
        train_ds = ds.skip(valid_size * 2)
        valid_ds = ds.take(valid_size * 2).skip(valid_size)
        test_ds = ds.take(valid_size)

        # SHUFFLE by images
        if shuffle:
            train_ds = train_ds.shuffle(buffer_size=shuffle_buffer_size)
            valid_ds = valid_ds.shuffle(buffer_size=shuffle_buffer_size)
            test_ds = test_ds.shuffle(buffer_size=shuffle_buffer_size)

        train_ds = train_ds.batch(BATCH_SIZE)
        valid_ds = valid_ds.batch(BATCH_SIZE)
        test_ds = test_ds.batch(BATCH_SIZE)
        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
        valid_ds = valid_ds.prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

        return train_ds, valid_ds, test_ds


def factory(files_list,
            dataset_size=DATASET_SIZE,
            out_shape=(193, 193),
            reshape_method=ImgReshape.RESIZE,
            shuffle=True):
    labeled_ds = files_list.map(
        lambda file_path: _process_path(file_path,
                                        out_shape=out_shape,
                                        reshape_method=reshape_method),
        num_parallel_calls=AUTOTUNE)

    return _prepare_for_training(labeled_ds, dataset_size=dataset_size, shuffle=shuffle)

