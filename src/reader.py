import nibabel as nib
import numpy as np
import logging
import os
from typing import Callable


# TODO: Could be refactor as TFRecord class
def nii_reader(path: str, default_shape: tuple = (256, 256, 166), ignore_shape: bool = True):
    # TODO: Could be extended to multiple formats
    # TODO: Could return objects of Nibel etc.
    # TODO: Optimalize reading, test different methods
    # https://simpleitk.readthedocs.io
    # https://nipy.org/nibael/
    # https://nilearn.github.io/
    assert os.path.isfile(path)
    img = nib.load(path)
    if img.shape != default_shape:
        logging.warning(f'Unexpected shape {img.shape}, default shape {default_shape}, file {path}')
        if ignore_shape: return None

    return np.squeeze(np.array(img.get_fdata()))


def nii_dir_generator(input_dir: str,
                      fname2label: Callable[[str], str] = None,
                      image_ext: str = "nii",
                      default_shape: tuple = (256, 256, 166),
                      ignore_shape: bool = False):
    for (dirpath, dirnames, filenames) in os.walk(input_dir):
        for f in filenames:
            if f.endswith(image_ext):
                f_path = os.path.join(dirpath, f)
                logging.info(f'Read nii file from {f_path}')
                img = nii_reader(f_path, default_shape=default_shape, ignore_shape=ignore_shape)
                if fname2label:
                    yield fname2label(f), img
                else:
                    yield f, img