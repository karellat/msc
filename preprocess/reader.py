import os
from nilearn import image
import numpy as np
import logging

def read_nii_arrays(input_dir, image_ext="nii", default_shape=(256, 256, 166)):
    # easy option
    nii_images = []
    for (dirpath, dirnames, filenames) in os.walk(input_dir):
        for f in filenames:
            if f.endswith(image_ext):
                logging.info("Read nii file from {}.".format(os.path.join(dirpath, f)))
                img = np.squeeze(np.array(image.load_img(os.path.join(dirpath, f)).get_data()))
                if img.shape != default_shape:
                    logging.warning("Expected {} shape but {} found in {}".format(default_shape, img.shape , os.path.join(dirpath, f))) 
                nii_images.append(img)
    return np.array(nii_images)
