from deep_mri.dataset import DEFAULT_PATH, DEFAULT_2D_PATH, DEFAULT_CSV_PATH, CLASS_NAMES
from deep_mri.dataset import dataset_3d, dataset_2d, dataset_encoder
from deep_mri.dataset.dataset import get_train_valid_files, _get_image_group


def dataset_factory(dataset_name,
                    train_filter_first_screen,
                    valid_filter_first_screen,
                    data_path='default',
                    data_csv_path='default',
                    dropping_group=None,
                    group_folder=None,
                    **dataset_args):
    data_csv_path = DEFAULT_CSV_PATH if data_csv_path is None or data_csv_path == 'default' else data_csv_path
    class_names = CLASS_NAMES if dropping_group is None else CLASS_NAMES.remove(dropping_group)
    group_folder = -3 if group_folder is None else group_folder
    if dataset_name.lower() == "3d":
        data_path = DEFAULT_PATH if data_path is None or data_path == 'default' else data_path
        train_files, train_targets, valid_files, valid_targets = get_train_valid_files(path=data_path,
                                                                                       csv_path=data_csv_path,
                                                                                       dropping_group=dropping_group,
                                                                                       train_filter_first_screen=train_filter_first_screen,
                                                                                       valid_filter_first_screen=valid_filter_first_screen,
                                                                                       img_group_fnc=lambda
                                                                                           x: _get_image_group(x,
                                                                                                               group_folder))
        assert len(train_files) > 0
        assert len(valid_files) > 0
        return dataset_3d.factory(train_files, train_targets, valid_files, valid_targets, class_names, **dataset_args)
    elif dataset_name.lower() == "2d":
        data_path = DEFAULT_2D_PATH if data_path is None or data_path == 'default' else data_path
        train_files, train_targets, valid_files, valid_targets = get_train_valid_files(path=data_path,
                                                                                       csv_path=data_csv_path,
                                                                                       train_filter_first_screen=train_filter_first_screen,
                                                                                       valid_filter_first_screen=valid_filter_first_screen,
                                                                                       dropping_group=dropping_group,
                                                                                       img_group_fnc=lambda
                                                                                           x: _get_image_group(x,
                                                                                                               group_folder))
        assert len(train_files) > 0
        assert len(valid_files) > 0
        return dataset_2d.factory(train_files, train_targets, valid_files, valid_targets, class_names, **dataset_args)
    elif dataset_name.lower() == "encoder":
        data_path = DEFAULT_PATH if data_path is None or data_path == 'default' else data_path
        train_files, _, valid_files, _ = get_train_valid_files(path=data_path,
                                                               csv_path=data_csv_path,
                                                               train_filter_first_screen=train_filter_first_screen,
                                                               valid_filter_first_screen=valid_filter_first_screen,
                                                               dropping_group=dropping_group,
                                                               img_group_fnc=lambda
                                                                   x: _get_image_group(x, group_folder))
        assert len(train_files) > 0
        assert len(valid_files) > 0
        return dataset_encoder.factory(train_files, valid_files, **dataset_args)
    else:
        raise Exception(f"Unknown type of dataset {dataset_name}")
