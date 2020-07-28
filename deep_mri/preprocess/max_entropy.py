import os
from nilearn import image, plotting
import numpy as np
from scipy import stats
from os import walk
import tqdm
from scipy.stats import rv_histogram
import imageio


def calc_entropy(array):
    hist = np.histogram(array)
    rv = rv_histogram(hist)
    return float(rv.entropy())


def max_entropy_slices(img, index):
    slices = img.reshape((img.shape[0], img.shape[1] * img.shape[2]))
    entropies = [calc_entropy(slices[i]) for i in range(256)]
    means = [np.mean(slices[i]) for i in range(256)]
    max_entropy = np.argsort(entropies)
    max_mean = np.argsort(means)
    entropy_mean_difference = len(set(max_entropy[index:]) - set(max_mean[index:]))
    return entropy_mean_difference, max_entropy[index:]


diff = 0
pbar = tqdm.tqdm()
for (dirpath, dirnames, filenames) in walk("data/OASIS2"):
    for i in filenames:
        if i.endswith(".img"):
            name = dirpath + "/" + i
            group = dirpath.split("/")[1].split("_")[-1][-1]
            session = dirpath.split("/")[-2]
            date = dirpath.split("/")[-1]
            img = np.squeeze(image.smooth_img(name, None).get_data())
            output_path = "/mnt/d/MRI"
            n = name.split("/")[2] + "_" + name.split("/")[3]
            if img.shape != (256, 256, 128):
                continue
            diff_mean_voxel, slices = max_entropy_slices(img, -5)
            diff += diff_mean_voxel
            for s in slices:
                new_img = img[s]
                new_name = n + "_" + str(s)
                np.save(os.path.join(output_path, new_name + ".npy"), new_img)
                imageio.imsave(os.path.join(output_path, new_name + ".jpg"), new_img)
            pbar.update(1)
    pbar.close()
print("Difference {}".format(diff))
