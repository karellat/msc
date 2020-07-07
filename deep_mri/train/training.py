import tensorflow as tf
from datetime import datetime
import logging
import os
import json

from deep_mri.model_zoo import model_factory
from deep_mri.dataset import dataset_factory


def run_train(path_to_config):
    # load config
    with open(path_to_config, 'r') as f:
        config = json.load(f)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

    assert ("filter_first_scan" not in config), "Deprecated options used in config"
    # Model setting
    assert ('model' in config) and ('model_args' in config)
    model_name = config['model']
    model_args = config['model_args']
    assert "callbacks" in config
    callbacks_names = config['callbacks']
    # Logs setting
    assert ("log_root" in config) and ("log_name" in config)
    log_root = config['log_root']
    log_name = config['log_name']
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
    # Training  setting
    assert "batch_size" in config
    batch_size = config['batch_size']
    assert "epochs" in config
    epochs = config["epochs"]
    assert "optimizer" in config
    optimizer = config["optimizer"]
    assert "loss" in config
    loss = config["loss"]
    assert "metrics" in config
    metrics = config['metrics']
    init_lr = config['init_lr'] if "init_lr" in config else 0.001

    if isinstance(model_name, list):
        assert isinstance(model_args, list)
        assert isinstance(log_name, list)
    else:
        model_name = [model_name]
        model_args = [model_args]
        log_name = [log_name]

    assert (len(model_name) == len(log_name)) & (len(model_args) == len(model_name))

    for m_name, m_args, log_n in zip(model_name, model_args, log_name):
        log_dir = os.path.join(log_root, log_n)
        model = model_factory(model_name=m_name, **m_args)
        logging.warning(f"Model name: {model.name}")
        model.summary(print_fn=logging.warning)

        callbacks = []
        if 'board' in callbacks_names:
            callbacks.append(
                tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                               histogram_freq=0,
                                               write_graph=False)
            )
        if 'save' in callbacks_names:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(os.path.join(log_dir, 'models'),
                                                   save_best_only=True)
            )
        if 'lr_poly' in callbacks_names:
            schedular = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=init_lr,
                                                                      decay_steps=epochs,
                                                                      end_learning_rate=init_lr/100)
            callbacks.append(tf.keras.callbacks.LearningRateScheduler(schedule=schedular))
        if 'early_stop' in callbacks_names:
            callbacks.append(tf.keras.callbacks.EarlyStopping(patience=10))

        dataset = dataset_factory(dataset_name=dataset_name,
                                  train_filter_first_screen=train_filter_first_screen,
                                  valid_filter_first_screen=valid_filter_first_screen,
                                  dropping_group=dropping_group,
                                  data_csv_path=data_csv_path,
                                  data_path=dataset_path,
                                  group_folder=group_folder,
                                  **dataset_args)
        logging.warning(f"Dataset: {dataset_name}")
        logging.warning(f"args: {dataset_args}")

        if len(dataset) == 2:
            train_ds, valid_ds = dataset
        elif len(dataset) == 3:
            train_ds, valid_ds, test_ds = dataset
        else:
            raise Exception("Unsupported number of datasets split")

        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)

        model.optimizer.learning_rate = init_lr

        model.fit(train_ds.batch(batch_size),
                  epochs=epochs,
                  validation_data=valid_ds.batch(batch_size),
                  verbose=1,
                  workers=40,
                  use_multiprocessing=True,
                  callbacks=callbacks
                  )
