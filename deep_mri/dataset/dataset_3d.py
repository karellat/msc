import tensorflow as tf
import os
import numpy as np
from nilearn.image import resample_img
import nibabel as nib
import logging
import random
from deep_mri.dataset.dataset import CLASS_NAMES

BUFFER_SIZE = 64
AUTOTUNE = tf.data.experimental.AUTOTUNE


def _get_label_tf(file_path, class_folder=3):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[class_folder] == CLASS_NAMES


def _get_label_str(file_path, class_folder=3):
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
    label = _get_label_tf(file_path, class_folder=3)
    img = _decode_img(file_path, normalize, downscale_ratio)
    return img, label


def _merge_items(dictionary):
    items = []
    for key in dictionary.keys():
        items += dictionary[key]
    return items


def factory(files_list,
            img_shape=(193, 229, 193, 1),
            downscale_ratio=1,
            normalize=True,
            shuffle=True,
            return_testset=False,
            seed=42):
    rnd = random.Random(seed)
    output_shape = np.ceil(np.array(img_shape) / downscale_ratio).astype(int)
    scans = {c: [] for c in CLASS_NAMES}
    for f in files_list:
        target = CLASS_NAMES[np.argmax(_get_label_str(f))]
        scans[target].append(f)

    groups_count = np.array([len(scans[key]) for key in scans.keys()])
    for count, group in zip(groups_count, scans.keys()):
        logging.info(f'{group.upper()} count: {count}')

    num_folds = 10
    folds_size = np.ceil(groups_count / num_folds).astype(int)
    # shuffle
    if shuffle:
        for k in scans.keys():
            random.shuffle(scans[k])

    if return_testset:
        train_files = {key: scans[key][fold_size * 2:] for key, fold_size in zip(scans.keys(), folds_size)}
        test_files = {key: scans[key][0:fold_size] for key, fold_size in zip(scans.keys(), folds_size)}
        valid_files = {key: scans[key][fold_size:fold_size * 2] for key, fold_size in zip(scans.keys(), folds_size)}
    else:
        train_files = {key: scans[key][fold_size:] for key, fold_size in zip(scans.keys(), folds_size)}
        valid_files = {key: scans[key][0:fold_size] for key, fold_size in zip(scans.keys(), folds_size)}

    train_files = _merge_items(train_files)
    valid_files = _merge_items(valid_files)

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

    if return_testset:
        test_files = _merge_items(test_files)
        if shuffle:
            rnd.shuffle(test_files)
        test_ds = tf.data.Dataset.from_generator(_generator,
                                                 output_types=(tf.float32, tf.bool),
                                                 output_shapes=(output_shape, (3,)),
                                                 args=[test_files, normalize, downscale_ratio])
        test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

        return train_ds, valid_ds, test_ds
    else:
        return train_ds, valid_ds
