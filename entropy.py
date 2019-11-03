from preprocess.normalization import normalize
from preprocess.reader import read_nii_arrays
from preprocess.analyzer import voxel_dist_operations
import pickle
import argparse

parser = argparse.ArgumentParser(description="Characteristics of images.")
parser.add_argument("--data", "-d", 
                    help="Root directory of data",
                    type=str,
                    required=True)
parser.add_argument("--output", "-o",
                    help="Output file",
                    default="data.pickle",
                    type=str)
parser.add_argument("--ext", "-e",
                    help="Image extension",
                    default="nii",
                    type=str)



args = parser.parse_args()

data  = read_nii_arrays(args.data, image_ext=args.ext, default_shape=(256, 256, 166))
norm_data = normalize(data)
result = voxel_dist_operations(norm_data)
with open(args.output, "wb") as f: 
    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    