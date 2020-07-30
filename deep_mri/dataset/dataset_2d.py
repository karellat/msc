import numpy as np
import tensorflow as tf
from deep_mri.dataset.dataset import _get_label_tf
import tensorflow_addons as tfa
import random as rnd


def _process_path(file_path, target, img_size, channels, class_names, transform):
    """
    Transform path to the image tensor and label tensor.

    Parameters
    ----------
    file_path : str
        Image file path
    target : str
        Name of the ADNI group
    img_size : int
        Resolution of square image
    channels : int
        Number of image channels
    class_names : list
        List of all the ADNI group names
    transform: str
        Name of the data augmentation 
        
    Returns
    -------
    tuple
        Image tensor and Label tensor
    """
    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img, channels=channels)
    if transform is not None:
        img = _aug_factory(transform, img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize_with_crop_or_pad(img,
                                           target_height=img_size,
                                           target_width=img_size)
    label = _get_label_tf(target, class_names)
    return img, label


def _generator(file_list, target_list, img_size, channels, class_names, shuffle=True, transform=None):
    """
    Generator of 2d images and its labels 
    
    Parameters
    ----------
    file_list : list
        List of image file paths
    target_list : list
        List of ADNI group names
    img_size : int
        Resolution of square image
    channels : int
        Number of image channels
    class_names : list
        List of all the ADNI group names
    transform: str
        Name of the data augmentation 

    Returns
    -------
    iterable
        Returns iterable generator yielding image tensors and label tensors
    """
    file_label_list = list(zip(file_list, target_list))
    if shuffle:
        rnd.shuffle(file_label_list)
    for file_name, target in file_label_list:
        # Return both transformed and normal img
        img, label = _process_path(file_name, target, img_size, channels, class_names, transform=None)
        yield img, label
        if transform is not None:
            img, label = _process_path(file_name, target, img_size, channels, class_names, transform)
            yield img, label


def _aug_factory(name, image):
    """
    Data augmentation of the image by method given by name

    Parameters
    ----------
    name : str
        Name of the data augmentation method
    image : tensor
        Image tensor
    Returns
    -------
    tensor
        Return the transformed image tensor

    """
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


def factory(train_files, train_targets, valid_files, valid_targets, class_names, img_size=193, channels=3, shuffle=True,
            transform=None):
    """
    Factory of the 2D datasets

    Parameters
    ----------
    train_files : list
        Image paths of training dataset
    train_targets : list
        Labels of the training dataset
    valid_files : list
        Image paths of validation dataset
    valid_targets : list
        Labels of the validation dataset
    class_names : list
        List of all the ADNI group names
    img_size : int
        Resolution of square image
    channels : int
        Number of image channels
    shuffle : bool
        True if shuffling images in the datasets
    transform: str
        Name of the data augmentation

    Returns
    -------
    tuple
        2D Training tensorflow dataset, 2D Validation tensorflow dataset
    """
    img_shape = np.array((img_size, img_size, channels)).astype(int)

    if transform is not None:
        train_ds = tf.data.Dataset.from_generator(_generator,
                                                  output_types=(tf.float32, tf.bool),
                                                  output_shapes=(img_shape, (len(class_names),)),
                                                  args=[train_files, train_targets, img_size, channels, class_names,
                                                        shuffle,
                                                        transform])
    else:
        train_ds = tf.data.Dataset.from_generator(_generator,
                                                  output_types=(tf.float32, tf.bool),
                                                  output_shapes=(img_shape, (len(class_names),)),
                                                  args=[train_files, train_targets, img_size, channels, class_names,
                                                        shuffle])
    valid_ds = tf.data.Dataset.from_generator(_generator,
                                              output_types=(tf.float32, tf.bool),
                                              output_shapes=(img_shape, (len(class_names),)),
                                              args=[valid_files, valid_targets, img_size, channels, class_names,
                                                    shuffle])

    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    valid_ds = valid_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_ds, valid_ds
