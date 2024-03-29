import os
from deep_mri.dataset import dataset_factory, CLASS_NAMES
import tensorflow as tf
import numpy as np

from deep_mri.model_zoo import model_factory
from .callbacks import SaveConfigCallback, DummyPredictorCallback, ConfusionMatrixCallback


def config_to_ds(config):
    """
    Create the dataset based on info from config

    Parameters
    ----------
    config : dict
        Config dictionary

    Returns
    -------
    tf.keras.Dataset

    """
    # Dataset setting
    assert ("dataset_path" in config) and ("train_filter_first_screen" in config) and (
            "valid_filter_first_screen" in config)
    assert ("dataset" in config) and ('dataset_args' in config)
    assert ("data_csv_path" in config)
    dataset_path = config['dataset_path']
    data_csv_path = config['data_csv_path']
    train_filter_first_screen = config['train_filter_first_screen']
    valid_filter_first_screen = config['valid_filter_first_screen']
    dataset_name = config['dataset']
    dataset_args = config['dataset_args']
    if 'group_folder' in config:
        group_folder = config['group_folder']
    else:
        group_folder = None
    if 'dropping_group' in config:
        dropping_group = config['dropping_group']
    else:
        dropping_group = None
    if 'shuffle_strategy' in config:
        shuffle_strategy = config['shuffle_strategy']
    else:
        shuffle_strategy = None
    train_ds, valid_ds = dataset_factory(dataset_name=dataset_name,
                                         train_filter_first_screen=train_filter_first_screen,
                                         valid_filter_first_screen=valid_filter_first_screen,
                                         dropping_group=dropping_group,
                                         data_csv_path=data_csv_path,
                                         data_path=dataset_path,
                                         shuffle_strategy=shuffle_strategy,
                                         group_folder=group_folder,
                                         **dataset_args)

    return train_ds, valid_ds


def config_to_callbacks(config, train_ds, valid_ds):
    assert "callbacks" in config
    callbacks_names = config['callbacks'] if 'callbacks' in config else []
    # Logs setting
    assert ("log_root" in config) and ("log_name" in config)

    callbacks = []
    if 'board' in callbacks_names:
        log_root = config['log_root']
        log_name = config['log_name']
        log_dir = os.path.join(log_root, log_name)
        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                           histogram_freq=0,
                                           write_graph=False)
        )
    if 'save' in callbacks_names:
        log_root = config['log_root']
        log_name = config['log_name']
        log_dir = os.path.join(log_root, log_name)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(os.path.join(log_dir, 'models'),
                                               save_best_only=True)
        )
    if 'save_config' in callbacks_names:
        log_root = config['log_root']
        log_name = config['log_name']
        log_dir = os.path.join(log_root, log_name)
        callbacks.append(
            SaveConfigCallback(log_dir, config)
        )
    if 'dummy' in callbacks_names:
        callbacks.append(
            DummyPredictorCallback(train_ds, valid_ds)
        )
    if 'cm' in callbacks_names:
        log_root = config['log_root']
        log_name = config['log_name']
        log_dir = os.path.join(log_root, log_name)
        if 'dropping_group' in config:
            dropping_group = config['dropping_group']
            dropping_group = dropping_group.lower()
            class_names = np.array([g for g in CLASS_NAMES if g != dropping_group])
            assert dropping_group in CLASS_NAMES, f"Unknown group to drop {dropping_group}"
        else:
            class_names = CLASS_NAMES
        callbacks.append(
            ConfusionMatrixCallback(valid_ds, class_names, log_dir)
        )
    if 'lr_poly' in callbacks_names:
        assert "epochs" in config
        epochs = config["epochs"]
        init_lr = config['init_lr'] if "init_lr" in config else 0.001
        scheduler = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=init_lr,
                                                                  decay_steps=epochs,
                                                                  end_learning_rate=init_lr / 100)
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(schedule=scheduler))
    if 'early_stop' in callbacks_names:
        callbacks.append(tf.keras.callbacks.EarlyStopping(patience=10))

    return callbacks


def config_to_model(config):
    """
    Create the model based on info from config

    Parameters
    ----------
    config : dict
        Config dictionary

    Returns
    -------
    tf.keras.Model

    """
    assert "optimizer" in config
    optimizer = config["optimizer"]
    assert "loss" in config
    loss = config["loss"]
    assert "metrics" in config
    assert ('model' in config) and ('model_args' in config)
    model_name = config['model']
    model_args = config['model_args']

    metrics = config['metrics']
    init_lr = config['init_lr'] if "init_lr" in config else 0.001
    model = model_factory(model_name=model_name, **model_args)

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    model.optimizer.learning_rate = init_lr

    return model


def config_epochs(config):
    """
    Parse epochs from config dict

    Parameters
    ----------
    config : dict
        Config dictionary

    Returns
    -------
    int

    """
    return int(config_get_field(config, "epochs"))


def config_batch_size(config):
    """
    Parse batch size from config dict

    Parameters
    ----------
    config : dict
        Config dictionary

    Returns
    -------
    int

    """
    return int(config_get_field(config, "batch_size"))


def config_get_field(config, name):
    """
    Get field by given name from config dict
    Parameters
    ----------
    config : dict
        Config dictionary
    name : str
        Name of the config field

    """
    assert name in config, f"{name} not in config."
    return config[name]
