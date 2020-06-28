import os
import random
import numpy as np
import tensorflow as tf
from deep_mri.dataset.dataset import CLASS_NAMES


def _get_label_tf(file_path, class_folder=-4):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[class_folder] == CLASS_NAMES


def _process_path(file_path, img_size, channels):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img, channels=channels)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize_with_crop_or_pad(img,
                                           target_height=img_size,
                                           target_width=img_size)
    label = _get_label_tf(file_path)
    return img, label


def _generator(file_list, img_size, channels):
    for file_name in file_list:
        img, label = _process_path(file_name, img_size, channels)
        yield img, label


def factory(train_files, valid_files, img_size=193, channels=3, shuffle=True, seed=42):
    rnd = random.Random(seed)
    img_shape = np.array((img_size, img_size, channels)).astype(int)
    if shuffle:
        rnd.shuffle(train_files)
        rnd.shuffle(valid_files)

    train_ds = tf.data.Dataset.from_generator(_generator,
                                              output_types=(tf.float32, tf.bool),
                                              output_shapes=(img_shape, (3,)),
                                              args=[train_files, img_size, channels])
    valid_ds = tf.data.Dataset.from_generator(_generator,
                                              output_types=(tf.float32, tf.bool),
                                              output_shapes=(img_shape, (3,)),
                                              args=[valid_files, img_size, channels])

    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    valid_ds = valid_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_ds, valid_ds
