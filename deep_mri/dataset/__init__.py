"""
Module for the data operations such as loading, data augmentation, rescaling.

Creates tensorflow Dataset instances to provide simple usage in training cycle.
It can used either as part of the training module or as separated functions for loading the preprocessed datasets.
"""
import tensorflow as tf
import numpy as np

DEFAULT_PATH = '/ADNI/minc_beast/*/*/*.nii'
DEFAULT_CSV_PATH = '/ADNI/ADNI1_Complete_1Yr_1.5T_10_13_2019.csv'
DEFAULT_2D_PATH = '/ADNI/slice_minc/*/*/*/*.png'

CLASS_NAMES = np.array(['ad', 'mci', 'cn'])
AUTOTUNE = tf.data.experimental.AUTOTUNE

from .dataset_factory import dataset_factory
from .dataset import numpy_to_nibabel

__all__ = ["dataset_factory", "CLASS_NAMES", "AUTOTUNE", "DEFAULT_2D_PATH", "DEFAULT_PATH", "numpy_to_nibabel"]
