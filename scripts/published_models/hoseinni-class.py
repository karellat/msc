import json
with open('/home/karelto1/master_thesis/configs/hosseini-class.json', 'r') as f:
    config = json.load(f)


import tensorflow as tf
from deep_mri.model_zoo import model_factory
from deep_mri.train.config_parser import config_to_model, config_to_ds, config_epochs, config_batch_size, config_to_callbacks

subject_model = tf.keras.models.load_model("/home/karelto1/master_thesis/logs/final_3d/hosseini3_subjects/models/")

config['shuffle_strategy'] = 'subjects'
config['log_name'] += '_subjects'


subject_model_class = config_to_model(config)


for i in range(1, 41):
    subject_model_class.layers[i].set_weights(subject_model.layers[i].get_weights())

train_ds, valid_ds = config_to_ds(config)
callbacks = config_to_callbacks(config, train_ds, valid_ds)

batch_size = config_batch_size(config)
epochs = config_epochs(config)

subject_model_class.fit(train_ds.batch(batch_size),
          epochs=epochs,
          validation_data=valid_ds.batch(batch_size),
          verbose=1,
          workers=40,
          use_multiprocessing=True,
          callbacks=callbacks)

