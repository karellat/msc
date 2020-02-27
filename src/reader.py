import nibabel as nib
import numpy as np
import logging
import os
from typing import Callable
from scipy.ndimage import zoom

# TODO: Could be refactor as TFRecord class
def nii_reader(path: str, default_shape: tuple = (256, 256, 166), ignore_shape: bool = True, as_numpy: bool = True):
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
    
    if as_numpy:
        return np.squeeze(np.array(img.get_fdata()))
    else:
        return img

def nii_dir_generator(input_dir: str,
                      fname2label: Callable[[str], str] = None,
                      image_ext: str = "nii",
                      default_shape: tuple = (256, 256, 166),
                      ignore_shape: bool = False,
                      as_numpy: bool = True):
    for (dirpath, dirnames, filenames) in os.walk(input_dir):
        for f in filenames:
            if f.endswith(image_ext):
                f_path = os.path.join(dirpath, f)
                logging.info(f'Read nii file from {f_path}')
                img = nii_reader(f_path, default_shape=default_shape, ignore_shape=ignore_shape, as_numpy=as_numpy)
                if fname2label:
                    yield fname2label(f), img
                else:
                    yield f, img
                    
def img_to_shape(img: np.ndarray , new_shape: tuple = (110, 110, 110), mode :str = 'constant'):
    origin_shape = np.array(img.shape)
    new_shape = np.array(new_shape)
    
    return  zoom(input=img, zoom=new_shape/origin_shape, mode=mode)

def expand_channel_dim(img, expected_shape=3, export_type='float32'):
    assert len(img.shape)
    assert isinstance(img, np.ndarray)
    return img.reshape((*img.shape,1)).astype(export_type)

def max_entropy_slice_generator(img, n=32, normalizer=None):
    assert normalizer is not None
    nimg = normalizer(img)
    ent = np.zeros(nimg.shape[0])
    for i in range(nimg.shape[0]): 
        ent[i] = entropy.img_entropy(nimg[i, :, :])
    ent = np.argsort(ent)
    for i in range(n):
        # -1 for last img
        yield nimg[ent[-i-1],:,:]
