import os
from deep_mri.dataset import dataset_factory
import tensorflow as tf
from deep_mri.model_zoo import model_factory


def config_to_ds(config):
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

    train_ds, valid_ds = dataset_factory(dataset_name=dataset_name,
                                         train_filter_first_screen=train_filter_first_screen,
                                         valid_filter_first_screen=valid_filter_first_screen,
                                         dropping_group=dropping_group,
                                         data_csv_path=data_csv_path,
                                         data_path=dataset_path,
                                         group_folder=group_folder,
                                         **dataset_args)

    return train_ds, valid_ds


def config_to_callbacks(config):
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
    if 'lr_poly' in callbacks_names:
        assert "epochs" in config
        epochs = config["epochs"]
        init_lr = config['init_lr'] if "init_lr" in config else 0.001
        schedular = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=init_lr,
                                                                  decay_steps=epochs,
                                                                  end_learning_rate=init_lr / 100)
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(schedule=schedular))
    if 'early_stop' in callbacks_names:
        callbacks.append(tf.keras.callbacks.EarlyStopping(patience=10))

    return callbacks


def config_to_model(config):
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


def config_epochs(config):
    return int(config_get_field(config, "epochs"))


def config_batch_size(config):
    return int(config_get_field(config, ""))


def config_get_field(config, name):
    assert name in config
    return config[name]
