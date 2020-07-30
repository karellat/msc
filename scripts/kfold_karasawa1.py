import glob
import pandas as pd
import random
from deep_mri.dataset.dataset import _get_image_group, _get_image_id, CLASS_NAMES
import deep_mri.dataset
from deep_mri.train.config_parser import config_to_ds, config_to_model, config_to_callbacks, config_batch_size, config_epochs

import tensorflow  as tf
print("Num GPUs Available: ", (tf.config.experimental.list_physical_devices('GPU')))

import json
with open('/home/karelto1/master_thesis/configs/karasawa.json', 'r') as f:
    config = json.load(f)

config['shuffle_strategy'] = 'random'
    
files_list = glob.glob(config['dataset_path'])
# meta info
df = pd.read_csv(config['data_csv_path'])
df = df.set_index('Image Data ID')
df['Group'] = df['Group'].str.lower()
meta_info = df[['Visit', 'Group', 'Subject']].to_dict('index')
subjects = {c: [] for c in CLASS_NAMES}
for f in files_list:
    image_id = _get_image_id(f)
    target = _get_image_group(f, -3)
    assert target == meta_info[image_id]['Group']
    subject = meta_info[image_id]['Subject']
    visit = meta_info[image_id]['Visit']
    if visit == 1:
        subjects[target].append(subject)
rnd = random.Random(42)
for _, group in subjects.items():
    rnd.shuffle(group)

from sklearn.model_selection import KFold
import numpy as np
def cross_validation_files(subjects): 
    ad = np.array(subjects['ad'])
    mci = np.array(subjects['mci'])
    cn = np.array(subjects['cn'])
    kf = KFold(n_splits=5)

    for ad_idx, mci_idx, cn_idx in zip(kf.split(ad), kf.split(mci), kf.split(cn)):
        ad_train, ad_valid = ad[ad_idx[0]], ad[ad_idx[1]]
        mci_train, mci_valid = mci[mci_idx[0]], mci[mci_idx[1]]
        cn_train, cn_valid = cn[cn_idx[0]], cn[cn_idx[1]]

        train_subjects = set(np.concatenate([ad_train, mci_train, cn_train]))
        valid_subjects = set(np.concatenate([ad_valid, mci_valid, cn_valid]))

        train_files, valid_files = [], []
        train_targets, valid_targets =  [], []
        for f in files_list:
            image_id = int(_get_image_id(f))
            subject = meta_info[image_id]['Subject']
            target = _get_image_group(f, -3)
            assert target == meta_info[image_id]['Group']
            if subject in train_subjects: 
                train_files.append(f)
                train_targets.append(target)
            elif subject in valid_subjects: 
                valid_files.append(f)
                valid_targets.append(target)
            else: 
                pass
                #print(f'No first visit {subject}')
        yield train_files, train_targets,  valid_files, valid_targets
        
from deep_mri.dataset import dataset_3d
histories = []
for train_files, train_targets, valid_files, valid_targets in cross_validation_files(subjects):
    train_ds, valid_ds = dataset_3d.factory(train_files, train_targets, valid_files, valid_targets, CLASS_NAMES, **config['dataset_args'])
    print(train_ds, valid_ds)
    batch_size = config_batch_size(config)
    model = config_to_model(config)
    epochs = 20 

    history = model.fit(train_ds.batch(batch_size),
              epochs=epochs,
              validation_data=valid_ds.batch(batch_size),
              verbose=1,
              workers=40,
              use_multiprocessing=True)
    
    histories.append(history)
    
    
json_dict = {}
for i, hist in enumerate(histories): 
    json_dict[i] = hist.history
print(json_dict)
with open("history_random.json", "w") as f: 
    json.dump(json_dict, f)
