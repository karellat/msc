import tensorflow as tf
import numpy as np
from deep_mri.dataset import AUTOTUNE
from deep_mri.dataset.dataset import _get_label_tf
import random as rnd
from fsl.utils.image import resample
from fsl.data.image import Image


def _decode_img(path, normalize, out_shape):
    path = str(path, 'utf-8')
    img = Image(path)
    if out_shape is not None:
        img, _ = resample.resample(img, out_shape)
    tensor = tf.convert_to_tensor(np.array(img.data), tf.float32)
    tensor = tf.expand_dims(tensor, -1)
    if normalize:
        tensor /= 255.0
    return tensor


def _generator(file_list, target_list, normalize, out_shape, class_names, shuffle):
    file_label_list = list(zip(file_list, target_list))
    if shuffle:
        rnd.shuffle(file_label_list)
    for file_name, target in file_label_list:
        img, label = _process_path(file_name, target, normalize, out_shape, class_names)
        yield (img, label)


def _process_path(file_path, target, normalize, out_shape, class_names):
    label = _get_label_tf(target, class_names)
    img = _decode_img(file_path, normalize, out_shape)
    return img, label


def factory(train_files,
            train_targets,
            valid_files,
            valid_targets,
            class_names,
            img_shape=(193, 229, 193, 1),
            downscale_ratio=1,
            output_shape=None,
            normalize=True,
            shuffle=True):
    if output_shape is None:
        output_shape = np.ceil(np.array(img_shape) / downscale_ratio).astype(int)

    train_ds = tf.data.Dataset.from_generator(_generator,
                                              output_types=(tf.float32, tf.bool),
                                              output_shapes=(output_shape, (len(class_names),)),
                                              args=[train_files, train_targets, normalize, output_shape[:-1],
                                                    class_names, shuffle])
    valid_ds = tf.data.Dataset.from_generator(_generator,
                                              output_types=(tf.float32, tf.bool),
                                              output_shapes=(output_shape, (len(class_names),)),
                                              args=[valid_files, valid_targets, normalize, output_shape[:-1],
                                                    class_names, shuffle])

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    valid_ds = valid_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, valid_ds


def _encoder_generator(file_list, normalize, out_shape, shuffle):
    if shuffle:
        rnd.shuffle(file_list)
    for file_name in file_list:
        img = _decode_img(file_name, normalize, out_shape)
        yield (img, img)


def encoder_factory(train_files,
                    valid_files,
                    output_shape=(193, 229, 193, 1),
                    normalize=True,
                    shuffle=True):
    output_shape = np.array(output_shape).astype(int)

    train_ds = tf.data.Dataset.from_generator(_encoder_generator,
                                              output_types=(tf.float32, tf.bool),
                                              output_shapes=(output_shape, output_shape),
                                              args=[train_files, normalize, output_shape[:-1], shuffle])
    valid_ds = tf.data.Dataset.from_generator(_encoder_generator,
                                              output_types=(tf.float32, tf.bool),
                                              output_shapes=(output_shape, output_shape),
                                              args=[valid_files, normalize, output_shape[:-1], shuffle])

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    valid_ds = valid_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, valid_ds
