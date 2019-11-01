import os
from nilearn import image
import numpy as np
import logging

def read_nii_arrays(input_dir, default_shape=(256, 256, 166)):
    # easy option
    image.load_img("*nii", wildcards=True)
    nii_images = []
    for (dirpath, dirnames, filenames) in os.walk(input_dir):
        for f in filenames:
            if f.endswith(".nii"):
                logging.info("Read nii file from {}.".format(os.path.join(dirpath, f)))
                img = image.load_img(os.path.join(dirpath, f))
                if img.shape != default_shape:
                    logging.warning("Expected {} shape but {} found in {}".format(img.shape, default_shape, os.path.join(dirpath, f))) 
                nii_images.append(np.array(img.get_data()))
    return np.array(nii_images)
