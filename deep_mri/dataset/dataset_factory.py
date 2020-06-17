import tensorflow as tf
from deep_mri.dataset import DEFAULT_PATH, DEFAULT_2D_PATH
from deep_mri.dataset import dataset_3d, dataset_2d, dataset_encoder
from deep_mri.dataset.dataset import get_all_files


def dataset_factory(dataset_name, data_path, filter_first_scan, **dataset_args):
    if dataset_name.lower() == "3d":
        data_path = DEFAULT_PATH if data_path is None or data_path == 'default' else data_path
        files_list = get_all_files(path=data_path, filter_first_screen=filter_first_scan)
        return dataset_3d.factory(files_list, **dataset_args)
    elif dataset_name.lower() == "2d":
        if filter_first_scan:
            raise NotImplementedError()
        data_path = DEFAULT_2D_PATH if data_path is None or data_path == 'default' else data_path
        files_list = tf.data.Dataset.list_files(data_path)
        return dataset_2d.factory(files_list, **dataset_args)
    elif dataset_name.lower() == "encoder":
        data_path = DEFAULT_PATH if data_path is None or data_path == 'default' else data_path
        files_list = get_all_files(path=data_path, filter_first_screen=filter_first_scan)
        return dataset_encoder.factory(files_list, **dataset_args)
    else:
        raise Exception(f"Unknown type of dataset {dataset_name}")
