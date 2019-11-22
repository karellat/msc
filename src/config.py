import pandas as pd
from adni import get_adni_group
# CONFIG
# TODO: Move to config file
# IMG
# ADNI
ADNI_DF = pd.read_csv("ADNI1_Complete_1Yr_1.5T_11_21_2019.csv")
# READING
IMG_PATH = 'data'
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
T_BATCH_SIZE = 2
T_EPOCHS = 10
T_LOGS = 'logs'
T_CHECKPOINT = 'checkpoints'