import numpy as np
import tensorflow as tf
import nibabel as nib
import random
from nilearn.image import resample_img

from deep_mri.dataset import AUTOTUNE
from deep_mri.dataset.dataset import load_files_to_dataset


def _get_3d_boxes(img_array, N, box_size=5, max_tries=100, include_zeros=True):
    """
    Extract the cubes(boxes) from the 3D image

    Parameters
    ----------
    img_array : numpy.array
        Image encoded into numpy.array
    N : int
        Number of the desired boxes
    box_size : int
        Size of one side cube
    max_tries : int
        How many times iterate to get non zero cube
    include_zeros : bool
        Add cube of all zeros

    Returns
    -------
    list
        List of generated cubes
    """
    assert len(img_array.shape) == 3
    default_shape = img_array.shape
    boxes = []
    r = range(N - 1) if include_zeros else range(N)
    for _ in r:
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
    if include_zeros:
        boxes.append(np.zeros((box_size, box_size, box_size)))
    return boxes


def _generator(files_list, normalize, box_size, boxes_per_img, downscale_ratio, include_zeros=True):
    """
    Wraps the file list and cube extractor into a generator

    Parameters
    ----------
    files_list : list
        List of image file paths
    normalize: bool
        Transform the image voxel from 0..255 to 0..1
    box_size : int
        Size of one side cube
    boxes_per_img : int
        Number of the desired boxes
    downscale_ratio : float
        Desired output shape given by denominator
    include_zeros : bool
        Add cube of all zeros

    Returns
    -------
    iterable
        Returns iterable generator yielding pair of the same image cube
    """
    for file_name in files_list:
        img = nib.load(file_name)
        if downscale_ratio is not None and downscale_ratio != 1.0:
            img = resample_img(img, target_affine=np.eye(3) * downscale_ratio)
        boxes = _get_3d_boxes(img.get_fdata(), boxes_per_img, box_size)
        for box in boxes:
            tensor = tf.convert_to_tensor(box, tf.float32)
            tensor = tf.expand_dims(tensor, -1)
            if normalize:
                tensor /= 255.0
            yield (tensor, tensor)


def factory(train_files, valid_files, normalize=True, box_size=5, downscale_ratio=None, boxes_per_img=100,
            include_zeros=True):
    """
    Factory of the 3D encoder dataset

    Parameters
    ----------
    train_files : list
        Image paths of training dataset
    valid_files : list
        Image paths of validation dataset
    normalize: bool
        Transform the image voxel from 0..255 to 0..1
    box_size : int
        Size of one side cube
    downscale_ratio : float
        Desired output shape given by denominator
    boxes_per_img : int
        Number of the desired boxes
    include_zeros : bool
        Add cube of all zeros

    Returns
    -------
    tuple
        3D encoder training tensorflow dataset, 3D encoder validation tensorflow dataset
    """
    train_ds = load_files_to_dataset(train_files, len(train_files) * boxes_per_img, _generator, normalize=normalize,
                                     box_size=box_size, downscale_ratio=downscale_ratio, boxes_per_img=boxes_per_img,
                                     include_zeros=include_zeros)
    valid_ds = load_files_to_dataset(valid_files, len(valid_files) * boxes_per_img, _generator, normalize=normalize,
                                     box_size=box_size, downscale_ratio=downscale_ratio, boxes_per_img=boxes_per_img,
                                     include_zeros=include_zeros)

    train_ds = train_ds.prefetch(AUTOTUNE)
    valid_ds = valid_ds.prefetch(AUTOTUNE)

    return train_ds, valid_ds
