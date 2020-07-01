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
        img = _aug_factory(transform, img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize_with_crop_or_pad(img,
                                           target_height=img_size,
                                           target_width=img_size)
    label = _get_label_tf(file_path, class_names)
    return img, label


def _generator(file_list, img_size, channels, class_names, transform):
    for file_name in file_list:
        # Return both transformed and normal img
        img, label = _process_path(file_name, img_size, channels, class_names, transform=None)
        yield img, label
        if transform is not None:
            img, label = _process_path(file_name, img_size, channels, class_names, transform)
            yield img, label


def _aug_factory(name, image):
    # TODO: Remove fixed values
    name = str(name.decode()).lower()
    if name == 'saturation':
        return tf.image.adjust_saturation(image, 2)
    elif name == 'brightness':
        return tf.image.adjust_brightness(image, 0.1)
    elif name == 'blur':
        return tfa.image.gaussian_filter2d(image)
    elif name == 'mean':
        return tfa.image.mean_filter2d(image)
    elif name == 'median':
        return tfa.image.median_filter2d(image)
    elif name == 'contrast_up':
        return tf.image.adjust_contrast(image, 1.2)
    elif name == 'contrast_down':
        return tf.image.adjust_contrast(image, 0.8)
    elif name == 'crop':
        return tf.image.resize(
            tf.image.random_crop(image, size=tf.constant([96, 96, 3])),
            image.shape[:-1]) / 255
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

    train_ds = tf.data.Dataset.from_generator(_generator,
                                              output_types=(tf.float32, tf.bool),
                                              output_shapes=(img_shape, (3,)),
                                              args=[train_files, img_size, channels, class_names, transform])
    valid_ds = tf.data.Dataset.from_generator(_generator,
                                              output_types=(tf.float32, tf.bool),
                                              output_shapes=(img_shape, (3,)),
                                              args=[valid_files, img_size, channels, class_names])

    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    valid_ds = valid_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_ds, valid_ds
