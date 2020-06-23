import tensorflow as tf
import os
import numpy as np
from nilearn.image import resample_img
import nibabel as nib
import logging
import random
from deep_mri.dataset import CLASS_NAMES

BUFFER_SIZE = 64
AUTOTUNE = tf.data.experimental.AUTOTUNE
DEFAULT_CLASS_FOLDER = -3


def _get_label_tf(file_path, class_folder=DEFAULT_CLASS_FOLDER):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[class_folder] == CLASS_NAMES


def _get_label_str(file_path, class_folder=DEFAULT_CLASS_FOLDER):
    parts = file_path.split(os.path.sep)
    return parts[class_folder] == CLASS_NAMES


def _decode_img(path, normalize, downscale_ratio):
    img = nib.load(path)
    if downscale_ratio is not None and downscale_ratio != 1:
        img = resample_img(img, target_affine=np.eye(3) * downscale_ratio)
    tensor = tf.convert_to_tensor(img.get_fdata(), tf.float32)
    tensor = tf.expand_dims(tensor, -1)
    if normalize:
        tensor /= 255.0
    return tensor


def _generator(file_list, normalize, downscale_ratio):
    for file_name in file_list:
        file_name = file_name.decode('utf-8')
        img, label = _process_path(file_name, normalize, downscale_ratio)
        yield (img, label)


def _process_path(file_path, normalize, downscale_ratio):
    label = _get_label_tf(file_path)
    img = _decode_img(file_path, normalize, downscale_ratio)
    return img, label


def _merge_items(dictionary):
    items = []
    for key in dictionary.keys():
        items += dictionary[key]
    return items


def factory(train_files,
            valid_files,
            img_shape=(193, 229, 193, 1),
            downscale_ratio=1,
            normalize=True,
            shuffle=True,
            seed=42):
    rnd = random.Random(seed)
    output_shape = np.ceil(np.array(img_shape) / downscale_ratio).astype(int)

    if shuffle:
        rnd.shuffle(train_files)
        rnd.shuffle(valid_files)

    train_ds = tf.data.Dataset.from_generator(_generator,
                                              output_types=(tf.float32, tf.bool),
                                              output_shapes=(output_shape, (3,)),
                                              args=[train_files, normalize, downscale_ratio])
    valid_ds = tf.data.Dataset.from_generator(_generator,
                                              output_types=(tf.float32, tf.bool),
                                              output_shapes=(output_shape, (3,)),
                                              args=[valid_files, normalize, downscale_ratio])

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    valid_ds = valid_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, valid_ds
