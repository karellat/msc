import tensorflow as tf
import numpy as np

DEFAULT_PATH = '/ADNI/minc_beast/*/*/*.nii'
DEFAULT_2D_PATH = '/ADNI/slice_minc/*/*/*/*.png'

CLASS_NAMES = np.array(['ad', 'mci', 'cn'])
AUTOTUNE = tf.data.experimental.AUTOTUNE

from .dataset import dataset_factory

__all__ = ["dataset_factory"]