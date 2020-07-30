import numpy as np

from deep_mri.dataset import DEFAULT_PATH, DEFAULT_2D_PATH, DEFAULT_CSV_PATH, CLASS_NAMES
from deep_mri.dataset import dataset_3d, dataset_2d, dataset_encoder
from deep_mri.dataset.dataset import ShuffleStrategy, get_train_valid_files


def dataset_factory(dataset_name,
                    train_filter_first_screen,
                    valid_filter_first_screen,
                    data_path='default',
                    data_csv_path='default',
                    dropping_group=None,
                    group_folder=None,
                    shuffle_strategy=None,
                    **dataset_args):
    """
    Factor for creating various datasets

    Parameters
    ----------
    dataset_name : str
        Format of the dataset [ 3d, 2d, encoder_boxes, encoder_full]
    train_filter_first_screen : bool
        True if include only the first scans in training dataset
    valid_filter_first_screen : bool
        True if include only the first scans in validation dataset
    data_path : str
        Wildcard path to images
    data_csv_path : str
        Path to csv file with ADNI information such groups etc.
    dropping_group : str
        Exclude one of the adni groups from dataset
    group_folder : str
        Order of the group folder in image path
    shuffle_strategy : ShuffleStrategy
        Shuffle by random or by subjects
    dataset_args : dict
        Additional arguments for the specific dataset factory

    Returns
    -------
    tuple
        Training tensorflow dataset, Validation tensorflow dataset
    """
    if dropping_group is not None:
        dropping_group = dropping_group.lower()
        class_names = np.array([g for g in CLASS_NAMES if g != dropping_group])
        assert dropping_group in CLASS_NAMES, f"Unknown group to drop {dropping_group}"
    else:
        class_names = CLASS_NAMES
    data_csv_path = DEFAULT_CSV_PATH if data_csv_path is None or data_csv_path == 'default' else data_csv_path
    if shuffle_strategy is None or shuffle_strategy.lower() == 'subjects':
        shuffle_strategy = ShuffleStrategy.SHUFFLE_SUBJECTS
    elif shuffle_strategy.lower() == 'random':
        shuffle_strategy = ShuffleStrategy.SHUFFLE_RANDOM
    else:
        raise Exception(f"Unknown shuffle strategy {shuffle_strategy}")
    group_folder = -3 if group_folder is None else group_folder
    if dataset_name.lower() == "3d":
        data_path = DEFAULT_PATH if data_path is None or data_path == 'default' else data_path
        train_files, train_targets, valid_files, valid_targets = get_train_valid_files(path=data_path,
                                                                                       csv_path=data_csv_path,
                                                                                       dropping_group=dropping_group,
                                                                                       train_filter_first_screen=train_filter_first_screen,
                                                                                       valid_filter_first_screen=valid_filter_first_screen,
                                                                                       shuffle_strategy=shuffle_strategy,
                                                                                       group_folder=group_folder)
        assert len(train_files) > 0
        assert len(valid_files) > 0
        return dataset_3d.factory(train_files, train_targets, valid_files, valid_targets, class_names, **dataset_args)
    elif dataset_name.lower() == "2d":
        data_path = DEFAULT_2D_PATH if data_path is None or data_path == 'default' else data_path
        train_files, train_targets, valid_files, valid_targets = get_train_valid_files(path=data_path,
                                                                                       csv_path=data_csv_path,
                                                                                       train_filter_first_screen=train_filter_first_screen,
                                                                                       valid_filter_first_screen=valid_filter_first_screen,
                                                                                       shuffle_strategy=shuffle_strategy,
                                                                                       dropping_group=dropping_group,
                                                                                       group_folder=group_folder)
        assert len(train_files) > 0
        assert len(valid_files) > 0
        return dataset_2d.factory(train_files, train_targets, valid_files, valid_targets, class_names, **dataset_args)
    elif dataset_name.lower() == "encoder_boxes":
        data_path = DEFAULT_PATH if data_path is None or data_path == 'default' else data_path
        train_files, _, valid_files, _ = get_train_valid_files(path=data_path,
                                                               csv_path=data_csv_path,
                                                               train_filter_first_screen=train_filter_first_screen,
                                                               valid_filter_first_screen=valid_filter_first_screen,
                                                               shuffle_strategy=shuffle_strategy,
                                                               dropping_group=dropping_group,
                                                               group_folder=group_folder)
        assert len(train_files) > 0
        assert len(valid_files) > 0
        return dataset_encoder.factory(train_files, valid_files, **dataset_args)
    elif dataset_name.lower() == "encoder_full":
        data_path = DEFAULT_PATH if data_path is None or data_path == 'default' else data_path
        train_files, _, valid_files, _ = get_train_valid_files(path=data_path,
                                                               csv_path=data_csv_path,
                                                               train_filter_first_screen=train_filter_first_screen,
                                                               valid_filter_first_screen=valid_filter_first_screen,
                                                               shuffle_strategy=shuffle_strategy,
                                                               dropping_group=dropping_group,
                                                               group_folder=group_folder)
        assert len(train_files) > 0
        assert len(valid_files) > 0
        return dataset_3d.encoder_factory(train_files, valid_files, **dataset_args)
    else:
        raise Exception(f"Unknown type of dataset {dataset_name}")
