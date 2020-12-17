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
    plt.grid()
    plt.show()


def show_images(images, labels=None):
    # if images.shape[0] > MAX_IMAGES_IN_FIGURE:
    #     images = images[:MAX_IMAGES_IN_FIGURE, :, :, :]
    #
    # edge_size = int(np.round(np.sqrt(images.shape[0])))

    plt.figure()

    data_iter = iter(images.batch(16).unbatch())

    edge_size = 4

    for i in range(edge_size):
        for j in range(edge_size):
            index = i * edge_size + j
            # if index > images.shape[0]:
            #     break

            data = next(data_iter)
            img = data[0].numpy()[:, :, 0]
            labels = data[1].numpy()

            plt.subplot(edge_size, edge_size, index + 1)
            plt.imshow(img, cmap='gray')

            plot_labels(labels)

    # plt.show()


def show_corrupted_images(images):
    data_iter = iter(images.batch(4).unbatch())

    edge_size = 2

    for k in range(2):
        plt.figure()
        for i in range(edge_size):
            for j in range(edge_size):
                index = i * edge_size + j

                if k == 1 and index == 3:
                    continue

                data = next(data_iter)
                img = data[0].numpy()[:, :, 0]
                labels = data[1].numpy()

                plt.subplot(edge_size, edge_size, index + 1)
                plt.imshow(img, cmap='gray')

                plot_labels(labels)


def show_images_2(images, labelss, predictions, indexes=None):
    edge_size_y = 3
    edge_size_x = 9
    figure_size = edge_size_x * edge_size_y

    iteration = 0

    plt.figure()

    for iteration in range(images.shape[0]):
        for i in range(edge_size_y):
            for j in range(edge_size_x):
                index = iteration * figure_size + i * edge_size_x + j
                figure_index = i * edge_size_x + j
                # if index > images.shape[0]:
                #     break

                img = images[index, :, :, :]
                labels = labelss[index, :]
                pred = predictions[index, :]

                ax = plt.subplot(edge_size_y, edge_size_x, figure_index + 1)
                plt.imshow(img, cmap='gray')

                plot_labels(labels, 'blue')

                anns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

                # for k in range(len(anns)):
                #     ax.annotate(anns[k], (labels[k * 2], labels[k * 2 + 1]))

                plot_labels(pred, 'red')

                # for k in range(len(anns)):
                #     ax.annotate(anns[k], (pred[k * 2], pred[k * 2 + 1]))

                # if indexes is not None:
                #     plt.title(str(indexes[index]))

        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()

        plt.show()


def plot_labels(labels, c='red'):
    if labels is not None:
        x_labels = np.array([item[item > 0] or None for item in labels[0::2]])
        y_labels = np.array([item[item > 0] or None for item in labels[1::2]])
        x_labels = x_labels[x_labels is not None]
        y_labels = y_labels[y_labels is not None]

        plt.scatter(x_labels, y_labels, c=c, s=5)


def show_scales_graph():
    scale_1 = np.load(PROGRAM_DATA_ROOT + 'scale_1.npy')
    scale_2 = np.load(PROGRAM_DATA_ROOT + 'scale_2.npy')
    scale_3 = np.load(PROGRAM_DATA_ROOT + 'scale_3.npy')
    scale_4 = np.load(PROGRAM_DATA_ROOT + 'scale_4.npy')
    scale_5 = np.load(PROGRAM_DATA_ROOT + 'scale_5.npy')
    scale_6 = np.load(PROGRAM_DATA_ROOT + 'scale_6.npy')

    plt.plot(scale_1[1, :])
    plt.plot(scale_2[1, :])
    plt.plot(scale_3[1, :])
    plt.plot(scale_4[1, :])
    plt.plot(scale_5[1, :])
    plt.plot(scale_6[1, :])
    plt.legend(["Scale 1", "Scale 2", "Scale 3", "Scale 4", "Scale 5", "Scale 6"])
    plt.show()


def show_image_by_index(images, labels, index):
    plt.imshow(images[index, :, :, :])
    plot_labels(labels[index, :])
