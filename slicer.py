from preprocess.reader import nii_img_generator, parse_adni_id, get_slices
import pickle
import argparse
import os
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description="Characteristics of images.")
parser.add_argument("--data", "-d",
                    help="Root directory of data",
                    type=str,
                    required=True)
parser.add_argument("--output_dir", "-o",
                    help="Output dir",
                    required=True,
                    type=str)
parser.add_argument("--ext", "-e",
                    help="Image extension",
                    default="nii",
                    type=str)
parser.add_argument("--slices", "-s", 
                    help="Pickle 3 arrays contains slicing indices",
                    default="max_entropy_slices.pickle",
                    required=True,
                    type=str)

args = parser.parse_args()
#TODO: Make as argument

default_shape = (256, 256, 166)
with open(args.slices,"rb") as f:
    slices = pickle.load(f)

assert os.path.isdir(args.output_dir)
os.mkdir(os.path.join(args.output_dir, "X"))
os.mkdir(os.path.join(args.output_dir, "Y"))
os.mkdir(os.path.join(args.output_dir, "Z"))

for filename, img in nii_img_generator(args.data,
                             image_ext=args.ext,
                             default_shape=default_shape,
                             ignore_shape=True):
    adni_id = parse_adni_id(filename)
    #TODO: Check normalization
    x_slices, y_slices, z_slices = get_slices(img, slices, default_shape)

    for s, i in zip(x_slices, slices[0]):
        s = (s * 255 / np.max(s)).astype('uint8')
        Image.fromarray(s).save(os.path.join(args.output_dir,"X", "X{}_{}.png".format(str(i), adni_id)))
    for s, i in zip(y_slices, slices[1]):
        s = (s * 255 / np.max(s)).astype('uint8')
        Image.fromarray(s).save(os.path.join(args.output_dir,"Y", "Y{}_{}.png".format(str(i), adni_id)))
    for s, i in zip(z_slices, slices[2]):
        s = (s * 255 / np.max(s)).astype('uint8')
        Image.fromarray(s).save(os.path.join(args.output_dir,"Z", "Z{}_{}.png".format(str(i), adni_id)))
