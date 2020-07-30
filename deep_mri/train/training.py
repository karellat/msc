import json
import logging
from .config_parser import config_to_ds, config_to_model, config_to_callbacks, config_batch_size, config_epochs


def run_train(path_to_config, overriding_config):
    """
    Basic training loop using only config
    Parameters
    ----------
    path_to_config : str
        Path to config file
    overriding_config : dict
        Names and values of overriding fields from config

    """
    # load config
    with open(path_to_config, 'r') as f:
        config = json.load(f)

    for key, value in overriding_config.items():
        logging.warning(f"Overriding config {key} to {value}")
        config[key] = value

    train_ds, valid_ds = config_to_ds(config)
    model = config_to_model(config)
    callbacks = config_to_callbacks(config, train_ds, valid_ds)
    batch_size = config_batch_size(config)
    epochs = config_epochs(config)

    model.fit(train_ds.batch(batch_size),
              epochs=epochs,
              validation_data=valid_ds.batch(batch_size),
              verbose=1,
              workers=40,
              use_multiprocessing=True,
              callbacks=callbacks)
