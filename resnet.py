from resnet3d import Resnet3DBuilder
import tensorflow as tf
from deep_mri.dataset.dataset_3d import get_3d_dataset

BATCH_SIZE = 16
DOWNSCALE_RATIO = 2.0
EPOCHS = 100


models = {
    'resnet_18': Resnet3DBuilder.build_resnet_18((97, 115, 97, 1), 3),
    'resnet_34': Resnet3DBuilder.build_resnet_34((97, 115, 97, 1), 3),
    'resnet_50': Resnet3DBuilder.build_resnet_50((97, 115, 97, 1), 3),
    'resnet_101': Resnet3DBuilder.build_resnet_101((97, 115, 97, 1), 3),
    'resnet_152': Resnet3DBuilder.build_resnet_152((97, 115, 97, 1), 3)
}

for model_name, model in models:
    train_ds, valid_ds, test_ds = get_3d_dataset(downscale_ratio=DOWNSCALE_RATIO)

    log_dir = f"resnet/logs/-name{model_name}-b{BATCH_SIZE}"
    models_dir = log_dir + "/models"

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    model.fit(train_ds.batch(BATCH_SIZE),
              epochs=EPOCHS,
              validation_data=valid_ds.batch(BATCH_SIZE),
              verbose=1,
              workers=40,
              use_multiprocessing=True,
              callbacks=[tensorboard_callback])
