import logging
import tensorflow as tf
import json
import os
import sklearn
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import io


def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(8, 8))
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    ax = sns.heatmap(cm,
                     annot=True,
                     cmap=sns.color_palette("Blues"),
                     xticklabels=class_names,
                     yticklabels=class_names, )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    return figure


def plot_to_image(plot):
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(plot)
    buffer.seek(0)
    img = tf.image.decode_png(buffer.getvalue(), channels=4)
    img = tf.expand_dims(img, 0)
    return img


class SaveConfigCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, config):
        super().__init__()
        self.log_dir = log_dir
        self.config = config

    def on_train_begin(self, logs=None):
        config_path = os.path.join(self.log_dir, 'config.json')
        with open(config_path, 'w') as outfile:
            json.dump(self.config, outfile)


class DummyPredictorCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_ds, valid_ds):
        super().__init__()
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.metrics = {}

    def on_train_begin(self, logs=None):
        for name, ds in zip(['train', 'valid'], [self.train_ds, self.valid_ds]):
            labels = []
            for _, label in iter(ds):
                class_num = np.argmax(label)
                labels.append(class_num)
            labels = tf.stack(labels)
            y, idx, count = tf.unique_with_counts(labels)
            labels_sum = tf.math.reduce_sum(count)
            max_label = tf.math.argmax(count)
            dummy_acc = tf.math.reduce_max(count) / labels_sum
            logging.warning(f"{name} dataset")
            logging.warning(f"\tMaximum class {y[max_label]} : {count[max_label]}")
            logging.warning(f"\tMaxclass predictor accuracy {dummy_acc}")
            tf.summary.scalar(f"{name}_dummy", dummy_acc)


class ConfusionMatrixCallback(tf.keras.callbacks.Callback):

    def __init__(self, ds, class_names, log_dir, name="CM"):
        super().__init__()
        self.ds = ds
        self.class_names = class_names
        self.file_writer = tf.summary.create_file_writer(os.path.join(log_dir, name))
        self.name = name
        # Bool vector labels
        if len(ds.element_spec[1].shape) == 0:
            self.labels = np.array([label for _, label in iter(ds)])
        else:
            self.labels = np.array([np.argmax(label) for _, label in iter(ds)])

    def on_epoch_end(self, epoch, logs=None):
        pred_raw = self.model.predict(self.ds.batch(1))
        pred = np.argmax(pred_raw, axis=1)
        cm = sklearn.metrics.confusion_matrix(self.labels, pred)
        figure = plot_confusion_matrix(cm, self.class_names)
        img = plot_to_image(figure)
        with self.file_writer.as_default():
            tf.summary.image(self.name, img, step=epoch)