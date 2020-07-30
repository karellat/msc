from deep_mri.model_zoo.encoders import PayanEncoder
from deep_mri.dataset.dataset_encoder import DEFAULT_GENERATOR_ARGS, DEFAULT_PATH, get_encoder_dataset
import tensorflow as tf

args = DEFAULT_GENERATOR_ARGS
args['downscale_ratio'] = 3.0
args["boxes_per_img"] = 1000

def scheduler(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.1 * (10 - epoch))

lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath="./payan_encoder", save_best_only=True)



train_ds, valid_ds = get_encoder_dataset(DEFAULT_PATH, **args)
model = PayanEncoder()
model.compile(optimizer='adam', loss='mse', metrics='mse')

BATCH_SIZE=16
EPOCHS=100

model.fit(train_ds.batch(BATCH_SIZE),
          epochs=EPOCHS,
          validation_data=valid_ds.batch(BATCH_SIZE),
          callbacks=[lr_callback, tf.keras.callbacks.TensorBoard(log_dir="./payan_encoder"), model_checkpoint_callback]
          )
