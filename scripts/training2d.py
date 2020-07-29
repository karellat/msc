from deep_mri.dataset.dataset import get_datasets, CLASS_NAMES
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import io
from sklearn.metrics import confusion_matrix
from deep_mri.dataset.dataset import ImgReshape
import itertools


# Functions
def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def log_confusion_matrix(epoch, logs):
    # Use the model to predict the values from the validation dataset.
    test_pred_raw = model.predict(valid_ds)
    test_pred = np.argmax(test_pred_raw, axis=1)
    labels = np.concatenate(list(valid_ds.map(lambda img, label: label).as_numpy_iterator()))
    test_label = np.argmax(labels, axis=1)
    # Calculate the confusion matrix.
    cm = confusion_matrix(test_label, test_pred)
    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=CLASS_NAMES)
    cm_image = plot_to_image(figure)

    # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)


def lr_schedule(epoch):
    """
    Returns a custom learning rate that decreases as epochs progress.
    """
    learning_rate = 0.001
    if epoch > 70:
        learning_rate = learning_rate * (0.96 ** (epoch - 70))

    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate


# Preprocessing

# Iterate over preprecessing 
urls = [

    # "https://tfhub.dev/tensorflow/resnet_50/feature_vector/1",
    # "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/4",
    # "https://tfhub.dev/google/imagenet/resnet_v1_152/feature_vector/4",
    # "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4",
    # "https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/4",
    # "https://tfhub.dev/google/bit/m-r50x1/1",
    # "https://tfhub.dev/google/imagenet/pnasnet_large/feature_vector/4",
    # "https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1",
    # "https://tfhub.dev/tensorflow/efficientnet/b1/feature-vector/1",
    # "https://tfhub.dev/tensorflow/efficientnet/b2/feature-vector/1",
    # "https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1",
    "https://tfhub.dev/tensorflow/efficientnet/b3/feature-vector/1",
    "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",
    "https://tfhub.dev/tensorflow/efficientnet/b6/feature-vector/1",
    # "https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/feature_vector/4",
]

EPOCHS = 300
# TODO:  Solve smaller nets
non_compatible = ["https://tfhub.dev/google/imagenet/mobilenet_v2_050_96/feature_vector/4",
                  "https://tfhub.dev/google/imagenet/mobilenet_v2_075_96/feature_vector/4"]

out_shapes = [(193, 193), (229, 229), (331, 331), (128, 128), (224, 224)]
reshape_methods = [ImgReshape.RESIZE_CROP_PAD, ImgReshape.RESIZE, ImgReshape.RESIZE_PAD]

for url, out_shape, reshape_method in itertools.product(urls, out_shapes, reshape_methods):
    train_ds, valid_ds, test_ds = get_datasets(shuffle=False, out_shape=out_shape, reshape_method=reshape_method)
    try:
        feature_extractor_url = url
        feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                                 input_shape=(out_shape[0], out_shape[1], 3),
                                                 trainable=False)

        model = tf.keras.Sequential([
            feature_extractor_layer,
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(3, activation="softmax")
        ])

        model.summary()

        LOG_DIR = f"tf_shapes/logs/{url.split('/')[-4]}_{url.split('/')[-3]}_{out_shape}_{str(reshape_method).split('.')[-1]}"
        MODELS_DIR = LOG_DIR + "/models"
        file_writer_cm = tf.summary.create_file_writer(LOG_DIR + '/cm')

        file_writer = tf.summary.create_file_writer(LOG_DIR + "/metrics")
        file_writer.set_as_default()

        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)
        model_save_callback = tf.keras.callbacks.ModelCheckpoint(MODELS_DIR, save_best_only=True)
        cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
        earlystopping_callback = tf.keras.callbacks.EarlyStopping(patience=10)

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])


        model.fit(train_ds,
                  validation_data=valid_ds,
                  epochs=EPOCHS,
                  callbacks=[tensorboard_callback, model_save_callback, lr_callback, cm_callback])
    except Exception as e:
        print(e)
        continue
