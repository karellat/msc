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

from deep_mri.dataset import DEFAULT_PATH, CLASS_NAMES, DEFAULT_CSV_PATH

DEFAULT_CLASS_FOLDER = -3

def _merge_items(dictionary):
    items = []
    for key in dictionary.keys():
        items += dictionary[key]
    return items


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


def _get_image_id(name):
    return int(re.search('_image_id_([0-9]*)', name).group(1))


def _get_image_group(file_path, class_folder=DEFAULT_CLASS_FOLDER):
    parts = file_path.split(os.path.sep)
    assert np.sum(parts[class_folder] == CLASS_NAMES) == 1
    return parts[class_folder] == CLASS_NAMES


def get_train_valid_files(path=DEFAULT_PATH,
                          csv_path=DEFAULT_CSV_PATH,
                          train_filter_first_screen=True,
                          valid_filter_first_screen=False,
                          valid_train_ratio=0.2,
                          shuffle=False,
                          dropping_group=None,
                          im_id_fnc=_get_image_id,
                          img_group_fnc=_get_image_group):

    assert dropping_group not in CLASS_NAMES, f"Uknown group to drop {dropping_group}"
    files_list = glob.glob(path)
    # meta info
    df = pd.read_csv(csv_path)
    df = df.set_index('Image Data ID')
    df['Group'] = df['Group'].str.lower()
    meta_info = df[['Visit', 'Group', 'Subject']].to_dict('index')

    # Split into groups by subject id
    subjects = {c: [] for c in CLASS_NAMES}
    for f in files_list:
        image_id = int(im_id_fnc(f))
        target = CLASS_NAMES[np.argmax(img_group_fnc(f))]
        assert target == meta_info[image_id]['Group']
        subject = meta_info[image_id]['Subject']
        visit = meta_info[image_id]['Visit']
        if visit == 1:
            subjects[target].append(subject)

    # Shuffle
    rnd = random.Random(42)
    if shuffle:
        for group in subjects:
            rnd.shuffle(group)

    # Count groups
    groups_count = np.array([len(subjects[key]) for key in subjects.keys()])
    for count, group in zip(groups_count, subjects.keys()):
        logging.warning(f'{group.upper()} count: {count}')

    # Split Subjects into train valid groups
    valid_sizes = np.ceil(groups_count * valid_train_ratio).astype(int)
    train_subjects = {key: subjects[key][valid_size:] for key, valid_size in zip(subjects.keys(), valid_sizes)}
    valid_subjects = {key: subjects[key][:valid_size] for key, valid_size in zip(subjects.keys(), valid_sizes)}

    # Groups changed after visits
    train_subjects = _merge_items(train_subjects)
    valid_subjects = _merge_items(valid_subjects)

    train_files = []
    valid_files = []
    for f in files_list:
        image_id = int(im_id_fnc(f))
        target = CLASS_NAMES[np.argmax(img_group_fnc(f))]
        assert target == meta_info[image_id]['Group']
        subject = meta_info[image_id]['Subject']
        visit = meta_info[image_id]['Visit']
        # Drop unwanted groups
        if target == dropping_group:
            continue
        if subject in train_subjects:
            if train_filter_first_screen and visit != 1:
                continue
            train_files.append(f)
        elif subject in valid_subjects:
            if valid_filter_first_screen and visit != 1:
                continue
            valid_files.append(f)
        else:
            assert visit != 1, "None seen imgs"
            logging.error(f"Image {image_id} without first visit, subject {subject}")
            if not train_filter_first_screen:
                logging.error(f"{image_id} appending to train set")
                train_files.append(f)

    return train_files, valid_files
