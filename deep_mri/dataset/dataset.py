import tensorflow as tf
import numpy as np
import random
import os
import glob
import logging
from nibabel import Nifti2Image
from auto_tqdm import tqdm
import pandas as pd
import re

from deep_mri.dataset import DEFAULT_PATH, CLASS_NAMES


def _merge_items(dictionary):
    items = []
    for key in dictionary.keys():
        items += dictionary[key]
    return items


def get_label_str(file_path, class_folder=3):
    parts = file_path.split(os.path.sep)
    return parts[class_folder] == CLASS_NAMES


def train_valid_split_mri_files(files_list, seed=42, return_test=True, shuffle=False):
    rnd = random.Random(seed)
    scans = {c: [] for c in CLASS_NAMES}
    for f in files_list:
        target = CLASS_NAMES[np.argmax(get_label_str(f))]
        scans[target].append(f)

    groups_count = np.array([len(scans[key]) for key in scans.keys()])
    for count, group in zip(groups_count, scans.keys()):
        logging.info(f'{group.upper()} count: {count}')

    num_folds = 10
    folds_size = np.ceil(groups_count / num_folds).astype(int)
    # shuffle
    if shuffle:
        for k in scans.keys():
            random.shuffle(scans[k])

    train_files = {key: scans[key][fold_size * 2:] for key, fold_size in zip(scans.keys(), folds_size)}
    test_files = {key: scans[key][0:fold_size] for key, fold_size in zip(scans.keys(), folds_size)}
    valid_files = {key: scans[key][fold_size:fold_size * 2] for key, fold_size in zip(scans.keys(), folds_size)}

    train_files = _merge_items(train_files)
    test_files = _merge_items(test_files)
    valid_files = _merge_items(valid_files)

    if shuffle:
        rnd.shuffle(train_files)
        rnd.shuffle(test_files)
        rnd.shuffle(valid_files)
    if return_test:
        return train_files, valid_files, test_files
    else:
        return train_files, test_files + valid_files


def load_files_to_dataset(files_list, items_count, generator, **gen_arguments):
    input_arrays = []
    targets = []
    pbar = tqdm(total=items_count)
    gen = generator(files_list=files_list, **gen_arguments)
    for sample, target in gen:
        input_arrays.append(sample)
        targets.append(target)
        pbar.update(1)
    pbar.close()
    return tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(input_arrays), tf.convert_to_tensor(targets)))


def get_random_img_path(path=DEFAULT_PATH):
    files_list = glob.glob(path)
    return files_list[random.randint(0, len(files_list))]


def numpy_to_nibabel(numpy_array):
    return Nifti2Image(numpy_array, np.eye(4))


def get_all_files(path=DEFAULT_PATH, filter_first_screen=False):
    files_list = glob.glob(path)

    if filter_first_screen:
        first_screen = set(filter_first_image_id())
        return list(filter(lambda x: get_image_id(x) in first_screen, files_list))
    else:
        return files_list


def filter_first_image_id(csv_path='/ADNI/ADNI1_Complete_1Yr_1.5T_10_13_2019.csv'):
    df = pd.read_csv(csv_path)
    first_screen = df.groupby('Subject').agg({'Acq Date': 'min'}).reset_index()
    return pd.merge(df, first_screen, on=['Subject', 'Acq Date'], how='inner')['Image Data ID'].values


def get_image_id(name):
    return int(re.search('_image_id_([0-9]*)', name).group(1))


