import tensorflow as tf
import numpy as np
from nilearn.image import resample_img
import nibabel as nib
from deep_mri.dataset import AUTOTUNE
from deep_mri.dataset.dataset import _get_label_tf
# TODO: Remove
from deep_mri.dataset.dataset import _get_image_group, _get_image_id
import pandas as pd
import random as rnd

def _decode_img(path, normalize, downscale_ratio):
    img = nib.load(path)
    if downscale_ratio is not None and downscale_ratio != 1:
        img = resample_img(img, target_affine=np.eye(3) * downscale_ratio)
    tensor = tf.convert_to_tensor(img.get_fdata(), tf.float32)
    tensor = tf.expand_dims(tensor, -1)
    if normalize:
        tensor /= 255.0
    return tensor


def _generator(file_list, target_list, normalize, downscale_ratio, class_names):
    df = pd.read_csv('/home/karelto1/MRI/ADNI1_Complete_1Yr_1.5T_10_13_2019.csv')
    df = df.set_index('Image Data ID')
    df['Group'] = df['Group'].str.lower()
    meta_info = df[['Visit', 'Group', 'Subject']].to_dict('index')
    file_label_list = list(zip(file_list, target_list))
    rnd.shuffle(file_label_list)
    for file_name, target in file_label_list:

        file_name = str(file_name, 'utf-8')
        target = str(target, 'utf-8')
        assert target == _get_image_group(file_name, -3)
        assert meta_info[_get_image_id(file_list)]['Group'] == target
        file_name = file_name.decode('utf-8')
        img, label = _process_path(file_name, target, normalize, downscale_ratio, class_names)
        yield (img, label)


def _process_path(file_path, target, normalize, downscale_ratio, class_names):
    label = _get_label_tf(target, class_names)
    img = _decode_img(file_path, normalize, downscale_ratio)
    return img, label


def factory(train_files,
            train_targets,
            valid_files,
            valid_targets,
            class_names,
            img_shape=(193, 229, 193, 1),
            downscale_ratio=1,
            normalize=True):
    output_shape = np.ceil(np.array(img_shape) / downscale_ratio).astype(int)

    train_ds = tf.data.Dataset.from_generator(_generator,
                                              output_types=(tf.float32, tf.bool),
                                              output_shapes=(output_shape, (len(class_names),)),
                                              args=[train_files, train_targets, normalize, downscale_ratio,
                                                    class_names])
    valid_ds = tf.data.Dataset.from_generator(_generator,
                                              output_types=(tf.float32, tf.bool),
                                              output_shapes=(output_shape, (len(class_names),)),
                                              args=[valid_files, valid_targets, normalize, downscale_ratio,
                                                    class_names])

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    valid_ds = valid_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, valid_ds
