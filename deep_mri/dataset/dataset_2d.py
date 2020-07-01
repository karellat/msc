import os
import random
import numpy as np
import tensorflow as tf
from deep_mri.dataset.dataset import CLASS_NAMES
import tensorflow_addons as tfa


def _get_label_tf(file_path, class_names, class_folder=-4):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[class_folder] == class_names


def _process_path(file_path, img_size, channels, class_names, transform):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img, channels=channels)
    if transform is not None:
        img = transform
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize_with_crop_or_pad(img,
                                           target_height=img_size,
                                           target_width=img_size)
    label = _get_label_tf(file_path, class_names)
    return img, label


def _generator(file_list, img_size, channels, class_names, transform):
    for file_name in file_list:
        # Return both transformed and normal img
        img, label = _process_path(file_name, img_size, channels, class_names)
        yield img, label
        if transform is not None:
            img, label = _process_path(file_name, img_size, channels, class_names, transform)
            yield img, label


def _aug_factory(name):
    # TODO: Remove fixed values
    if name == 'saturation':
        return lambda x: tf.image.adjust_saturation(x, 2)
    elif name == 'brightness':
        return lambda x: tf.image.adjust_brightness(x, 0.1)
    elif name == 'blur':
        return tfa.image.gaussian_filter2d
    elif name == 'mean':
        return tfa.image.mean_filter2d
    elif name == 'median':
        return tfa.image.median_filter2d
    elif name == 'contrast_up':
        return lambda x: tf.image.adjust_contrast(x, 1.2)
    elif name == 'contrast_down':
        return lambda x: tf.image.adjust_contrast(x, 0.8)
    elif name == 'crop':
        return lambda x: tf.resize(
            tf.image.random_crop(x, size=tf.constant([96, 96, 3])),
            x.shape[:-1])
    else:
        raise Exception(f"Unknown data augmentation function {name}")


def factory(train_files, valid_files, dropping_group=None, img_size=193, channels=3, shuffle=True, transform=None,
            seed=42):
    class_names = CLASS_NAMES if dropping_group is None else CLASS_NAMES.remove(dropping_group)
    rnd = random.Random(seed)
    img_shape = np.array((img_size, img_size, channels)).astype(int)
    if shuffle:
        rnd.shuffle(train_files)
        rnd.shuffle(valid_files)

    transform_fnc = _aug_factory(transform)
    train_ds = tf.data.Dataset.from_generator(_generator,
                                              output_types=(tf.float32, tf.bool),
                                              output_shapes=(img_shape, (3,)),
                                              args=[train_files, img_size, channels, class_names, transform_fnc])
    valid_ds = tf.data.Dataset.from_generator(_generator,
                                              output_types=(tf.float32, tf.bool),
                                              output_shapes=(img_shape, (3,)),
                                              args=[valid_files, img_size, channels, class_names])

    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    valid_ds = valid_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_ds, valid_ds
