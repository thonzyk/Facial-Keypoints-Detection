from matplotlib import pyplot as plt
import math
import numpy as np
from .constants import *

MAX_IMAGES_IN_FIGURE = 16


def plot_train_history(history):
    plt.figure()
    plt.plot(history.history['loss'], label='trn RMSE')
    plt.plot(history.history['val_loss'], label='tst RMSE')
    plt.xlabel('Epoch')
    plt.legend(['train RMSE', 'test RMSE'])
    plt.show()


def show_images(images, labels=None):
    if images.shape[0] > MAX_IMAGES_IN_FIGURE:
        images = images[:MAX_IMAGES_IN_FIGURE, :, :, :]

    edge_size = int(np.round(np.sqrt(images.shape[0])))

    for i in range(edge_size):
        for j in range(edge_size):
            index = i * edge_size + j
            if index > images.shape[0]:
                break

            plt.subplot(edge_size, edge_size, index + 1)
            plt.imshow(images[index, :, :, :], cmap='gray')

            if labels is not None:
                x_labels = np.array([item[item > 0] or None for item in labels[index, 0::2]])
                y_labels = np.array([item[item > 0] or None for item in labels[index, 1::2]])
                x_labels = x_labels[x_labels is not None]
                y_labels = y_labels[y_labels is not None]

                plt.scatter(x_labels, y_labels, color='red', s=5)

    plt.show()
