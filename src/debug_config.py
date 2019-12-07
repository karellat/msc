import pandas as pd
from adni import get_adni_group
import os
# CONFIG
# TODO: Move to config file
# HOME_PATH='/storage/praha1/home/karellat/'
HOME_PATH='/mnt/c/Users/tomas/Desktop/master_thesis'
# IMG
# ADNI
ADNI_DF = pd.read_csv(os.path.join(HOME_PATH,"ADNI1_Baseline_3T_12_07_2019.csv"))
# READING
IMG_PATH = os.path.join(HOME_PATH,"examples")
IMG_EXT = 'nii'
IMG_SHAPE = (256, 256, 166)
MODEL_SHAPE = None
IMG_IGNORE_BAD_SHAPE = True
FNAME_TO_LABEL = lambda x: x
# NORMALIZATION
NORM_METHOD = 'MinMax'
NORM_RANGE = (0, 1)
# AUGMENTATION

# CLASS BALANCING

# TRAINING
T_BATCH_SIZE = 8
T_EPOCHS = 100
T_LOGS = os.path.join(HOME_PATH, 'logs')
T_CHECKPOINT = os.path.join(HOME_PATH, 'checkpoints')
