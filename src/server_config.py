import pandas as pd
from adni import get_adni_group
import os
from reader import expand_channel_dim, img_to_shape
# CONFIG
# TODO: Move to config file
HOME_PATH='/storage/praha1/home/karellat/'
# IMG
# ADNI
ADNI_DF = pd.read_csv(os.path.join(HOME_PATH, "ADNI1_Complete_1Yr_1.5T_11_04_2019.csv"))
# READING
IMG_PATH = os.path.join(HOME_PATH,"data", "ADNI")
IMG_EXT = 'nii'
IMG_SHAPE = (256, 256, 166)
MODEL_SHAPE = (110, 110, 110)
IMG_IGNORE_BAD_SHAPE = False
IMG_TRANSFORMS = [lambda x: img_to_shape(x, MODEL_SHAPE),
                      expand_channel_dim]
FNAME_TO_LABEL = lambda x: get_adni_group(x, ADNI_DF)
# NORMALIZATION
NORM_METHOD = 'MinMax'
NORM_RANGE = (0, 1)
# AUGMENTATION

# CLASS BALANCING
# MODEL
MODEL_NAME = "resnet"

# TRAINING
T_VALID_SIZE = 0.2
T_BATCH_SIZE = 8
T_EPOCHS = 100
T_LOGS = os.path.join(HOME_PATH, 'logs')
T_CHECKPOINT = os.path.join(HOME_PATH, 'checkpoints')
