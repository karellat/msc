import pandas as pd
from adni import get_adni_group
import os
# CONFIG
# TODO: Move to config file
HOME_PATH='/storage/praha1/home/karellat'
# IMG
# ADNI
ADNI_DF = pd.read_csv(os.path.join(HOME_PATH, "ADNI1_Complete_1Yr_1.5T_11_04_2019.csv"))
# READING
IMG_PATH = os.path.join(HOME_PATH,"data", "ADNI")
IMG_EXT = 'nii'
IMG_SHAPE = (256, 256, 166)
IMG_IGNORE_BAD_SHAPE = True
FNAME_TO_LABEL = lambda x: get_adni_group(x, ADNI_DF)
# NORMALIZATION
NORM_METHOD = 'MinMax'
NORM_RANGE = (0, 1)
# AUGMENTATION

# CLASS BALANCING

# TRAINING
T_BATCH_SIZE = 1
T_EPOCHS = 100
T_LOGS = os.path.join(HOME_PATH, 'logs')
T_CHECKPOINT = os.path.join(HOME_PATH, 'checkpoints')
