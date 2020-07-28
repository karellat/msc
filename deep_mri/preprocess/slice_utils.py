import matplotlib.pyplot as plt
import numpy as np
import math
from deep_mri.dataset.dataset import CLASS_NAMES


def get_not_matching(ds, model):
    not_matching_img = None
    not_matching_labels = None
    not_matching_pred = None
    for image_batch, label_batch in iter(ds):
        predicted = model.predict(image_batch)
        not_matching = np.argmax(predicted, axis=1) != np.argmax(label_batch, axis=1)
        if not_matching_img is None:
            not_matching_img = image_batch[not_matching].numpy()
            not_matching_labels = label_batch[not_matching].numpy()
            not_matching_pred = predicted[not_matching]
        else:
            not_matching_img = np.concatenate([not_matching_img, image_batch[not_matching].numpy()])
            not_matching_labels = np.concatenate([not_matching_labels, label_batch[not_matching].numpy()])
            not_matching_pred = np.concatenate([not_matching_pred, predicted[not_matching]])

    return not_matching_img, not_matching_labels, not_matching_pred


def show_batch(image_batch, label_batch, predicted_batch=None):
    if predicted_batch is not None:
        predicted_labels = np.argmax(predicted_batch, axis=1)
        matching = predicted_labels == np.argmax(label_batch, axis=1)
    correct_labels = np.argmax(label_batch, axis=1)
    img_per_row = 6
    num_imgs = image_batch.shape[0] / img_per_row
    num_rows = math.ceil(num_imgs)
    plt.figure(figsize=(20, image_batch.shape[0] * 0.625))

    for n in range(image_batch.shape[0]):
        ax = plt.subplot(num_rows, img_per_row, n + 1)
        plt.imshow(image_batch[n].squeeze())
        if predicted_batch is None:
            plt.title(CLASS_NAMES[correct_labels[n]])
        else:
            if matching[n]:
                plt.title(CLASS_NAMES[correct_labels[n]], color='green')
                plt.text(150, -8, f'{CLASS_NAMES[correct_labels[n]]} {predicted_batch[n][correct_labels[n]]:.2f}',
                         color='green')
            else:
                plt.title(f'{CLASS_NAMES[correct_labels[n]]}({CLASS_NAMES[predicted_labels[n]]})', color='red')
                plt.text(0,
                         -8,
                         f'{CLASS_NAMES[predicted_labels[n]]} {predicted_batch[n][predicted_labels[n]]:.2f}',
                         color='red')
                plt.text(150,
                         -8,
                         f'{CLASS_NAMES[correct_labels[n]]} {predicted_batch[n][correct_labels[n]]:.2f}', color='green')

        plt.axis('off')
    return plt
