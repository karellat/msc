from deep_mri.model_zoo.models_3d import payan_montana_model
from deep_mri.dataset.dataset import get_all_files
from deep_mri.dataset.dataset_3d import get_3d_dataset
EPOCHS = 100
BATCH_SIZE = 32
MODEL = payan_montana_model
MODEL_ARGS = {"input_shape": (97, 115, 97, 1)}
DATASET = get_3d_dataset
DATASET_ARGS = {'files_list' : get_all_files(filter_first_screen=True)}
LOG_DIR = "test"
CALLBACKS = ['']
