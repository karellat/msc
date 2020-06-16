import tensorflow as tf
from datetime import datetime

EPOCHS = 100
BATCH_SIZE = 32
MODEL = lambda x: x
MODEL_ARGS = {}
DATASET = lambda x: x
DATASET_ARGS = {}
LOG_DIR = ""

model = MODEL(**MODEL_ARGS)
dataset = DATASET(**DATASET_ARGS)

if len(dataset) == 2:
    train_ds, valid_ds = dataset
elif len(dataset) == 3:
    train_ds, valid_ds, test_ds = dataset
else:
    raise Exception("Unsupported number of datasets split")

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])

model.fit(train_ds.batch(BATCH_SIZE),
          epochs=EPOCHS,
          validation_data=valid_ds.batch(BATCH_SIZE),
          verbose=1,
          workers=40,
          use_multiprocessing=True,
          callbacks=[tensorboard_callback]
          )
