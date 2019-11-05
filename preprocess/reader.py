import os
from nilearn import image
import numpy as np
import logging
import re
from preprocess.normalization import normalize

def nii_img_generator(input_dir, image_ext="nii", default_shape=(256, 256, 166), ignore_shape=False):
    for (dirpath, dirnames, filenames) in os.walk(input_dir):
        for f in filenames:
            if f.endswith(image_ext):
                logging.info("Read nii file from {}.".format(os.path.join(dirpath, f)))
                img = np.squeeze(np.array(image.load_img(os.path.join(dirpath, f)).get_data()))
                if img.shape != default_shape:
                    logging.warning("Expected {} shape but {} found in {}".format(default_shape, img.shape , os.path.join(dirpath, f)))
                    if ignore_shape:
                        logging.warning("Ignoring {}".format(f))
                        continue
                yield f, img
                
def read_nii_arrays(input_dir, image_ext="nii", default_shape=(256, 256, 166), ignore_shape=False):
    nii_images = []
    for filename, img in nii_img_generator(input_dir, image_ext, default_shape, ignore_shape):
        nii_images.append(img)
    
    return np.array(nii_images)

def parse_adni_id(adni_img_name):
    assert adni_img_name.startswith("ADNI_")
    adni_id = re.findall(r"ADNI_([0-9]+_S_[0-9]+)_", adni_img_name)
    if len(adni_id) != 1:
        logging.error("Unknown subject ID: {}".format(adni_img_name))
    return adni_id[0]

def get_slices(img, slices_index, default_shape=(256, 256, 166)):
    assert img.shape == default_shape
    assert len(slices_index.shape) > 1
    assert slices_index.shape[0] == 3
    first_dim, second_dim, third_dim = [], [], []
    
    
    for ind in slices_index[0]:
        first_dim.append(img[ind,:,:])
    for ind in slices_index[1]:
        second_dim.append(img[:,ind,:])
    for ind in slices_index[2]:
        third_dim.append(img[:,:,ind])

    return first_dim, second_dim, third_dim

