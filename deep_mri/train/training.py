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

    assert ('model' in config) and ('model_args' in config)
    model_name = config['model']
    model_args = config['model_args']
    assert "callbacks" in config
    callbacks_names = config['callbacks']
    assert ("log_root" in config) and ("log_name" in config)
    log_root = config['log_root']
    log_name = f"{timestamp}-{config['log_name']}"
    log_dir = os.path.join(log_root, log_name)
    assert ("dataset_path" in config) and ("filter_first_scan" in config)
    assert ("dataset" in config) and ('dataset_args' in config)
    dataset_path = config['dataset_path']
    filter_first_scan = config['filter_first_scan']
    dataset_name = config['dataset']
    dataset_args = config['dataset_args']
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

    model = model_factory(model_name=model_name, **model_args)
    logging.warning(f"Model name: {model.name}")
    model.summary()

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
    if 'lr_poly':
        schedular = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=0.001,
                                                                  decay_steps=epochs,
                                                                  end_learning_rate=0.00001)
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(schedule=schedular))
    if 'early_stop':
        callbacks.append(tf.keras.callbacks.EarlyStopping(patience=10))

    dataset = dataset_factory(dataset_name=dataset_name,
                              filter_first_scan=filter_first_scan,
                              data_path=dataset_path,
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

    model.fit(train_ds.batch(batch_size),
              epochs=epochs,
              validation_data=valid_ds.batch(batch_size),
              verbose=1,
              workers=40,
              use_multiprocessing=True,
              callbacks=callbacks
              )
