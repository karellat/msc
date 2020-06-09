from deep_mri.dataset.dataset import train_valid_split_mri_files, load_files_to_dataset, DEFAULT_PATH, AUTOTUNE
import numpy as np
import tensorflow as tf
import nibabel as nib
import random
from nilearn.image import resample_img

DEFAULT_GENERATOR_ARGS = {
    "normalize": True,
    "box_size": 5,
    "downscale_ratio": None,
    "boxes_per_img": 100,
}


def get_3d_boxes(img_array, N, box_size=5, max_tries=100):
    assert len(img_array.shape) == 3
    default_shape = img_array.shape
    boxes = []
    for _ in range(N - 1):
        box = np.zeros((box_size, box_size, box_size, 1))
        tries = 0
        while np.count_nonzero(box) == 0:
            if tries > max_tries:
                raise Exception("Input image can be all zeros, reached max iteration")
            tries += 1
            x = random.randint(0, default_shape[0] - box_size - 1)
            y = random.randint(0, default_shape[1] - box_size - 1)
            z = random.randint(0, default_shape[2] - box_size - 1)
            box = img_array[x:x + box_size, y:y + box_size, z:z + box_size]
        boxes.append(box)
    # Zero matrix 
    boxes.append(np.zeros((box_size, box_size, box_size)))
    return boxes


def _generator(files_list, normalize, box_size, boxes_per_img, downscale_ratio):
    for file_name in files_list:
        img = nib.load(file_name)
        if downscale_ratio is not None and downscale_ratio != 1.0:
            img = resample_img(img, target_affine=np.eye(3) * downscale_ratio)
        boxes = get_3d_boxes(img.get_fdata(), boxes_per_img, box_size)
        for box in boxes:
            tensor = tf.convert_to_tensor(box, tf.float32)
            tensor = tf.expand_dims(tensor, -1)
            if normalize:
                tensor /= 255.0
            yield (tensor, tensor)


def get_encoder_dataset(path=DEFAULT_PATH, **gen_args):
    train_files, valid_files = train_valid_split_mri_files(return_test=False)
    train_ds = load_files_to_dataset(train_files, len(train_files) * gen_args['boxes_per_img'], _generator, **gen_args)
    valid_ds = load_files_to_dataset(valid_files, len(valid_files) * gen_args['boxes_per_img'], _generator, **gen_args)

    train_ds = train_ds.prefetch(AUTOTUNE)
    valid_ds = valid_ds.prefetch(AUTOTUNE)

    return train_ds, valid_ds
