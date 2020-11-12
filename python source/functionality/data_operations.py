import pandas as pd
import numpy as np
import math
from .constants import *
from matplotlib import pyplot as plt
from .models import *


def preprocess_data(csv_file):
    """"""
    x_train, x_val, x_test, y_train, y_val, y_test = load_train_data(csv_file)

    np.save(DATA_ROOT + 'x_train.npy', x_train)
    np.save(DATA_ROOT + 'x_val.npy', x_val)
    np.save(DATA_ROOT + 'x_test.npy', x_test)
    np.save(DATA_ROOT + 'y_train.npy', y_train)
    np.save(DATA_ROOT + 'y_val.npy', y_val)
    np.save(DATA_ROOT + 'y_test.npy', y_test)


def load_prepared_data(directory):
    """"""
    x_train = np.load(directory + 'x_train.npy')
    x_val = np.load(directory + 'x_val.npy')
    x_test = np.load(directory + 'x_test.npy')
    y_train = np.load(directory + 'y_train.npy', allow_pickle=True)
    y_val = np.load(directory + 'y_val.npy', allow_pickle=True)
    y_test = np.load(directory + 'y_test.npy', allow_pickle=True)
    return x_train, x_val, x_test, y_train, y_val, y_test


def transform_images(images):
    # Parse pixel values from the string
    # - memory consuming operation (optimize in case of memory problems)
    images = np.flip(np.array([pixels.split() for pixels in images]))
    images = images.astype('float32')

    image_size = int(round(math.sqrt(images.shape[-1])))

    images = images.reshape((images.shape[0], image_size, image_size))
    images = np.rot90(images, 2, axes=(1, 2))
    images = np.divide(images, 255)
    images = np.expand_dims(images, axis=-1)

    return images


def load_train_data(csv_file):
    """"""
    data = pd.read_csv(csv_file)
    data = data.to_numpy()

    np.random.shuffle(data)

    x = data[:, -1]
    y = data[:, :-1]

    x = transform_images(x)

    x_train, x_val, x_test = split_data(x)
    y_train, y_val, y_test = split_data(y)

    return x_train, x_val, x_test, y_train, y_val, y_test


def load_test_data(csv_file="test.csv"):
    """"""
    data = pd.read_csv(DATA_ROOT + csv_file)
    data = data.to_numpy()

    ids = data[:, 0]
    images = data[:, 1]

    images = transform_images(images)

    return ids, images

def load_subscription_mapping(csv_file="IdLookupTable.csv"):
    """"""
    data = pd.read_csv(DATA_ROOT + csv_file)

    mapping = dict()

    for index, row in data.iterrows():
        key = str(row['ImageId']) + '_' + row['FeatureName']
        mapping[key] = int(row["RowId"])

    return mapping


def split_data(data, train_portion=0.9, val_portion=None):
    """"""
    if not val_portion:
        val_portion = 1 - train_portion
        test_portion = float(0)
    else:
        test_portion = 1 - train_portion - val_portion

    # Check if sum of portions is 1
    max_err = pow(10, -9)
    assert abs((train_portion + val_portion + test_portion) - 1) < max_err

    split_index_1 = round(len(data[:, 0]) * train_portion)
    split_index_2 = round(len(data[:, 0]) * (train_portion + val_portion))

    # TODO: more sophisticated solution
    if len(data.shape) == 3:
        train_data = data[0:split_index_1, :, :]
        val_data = data[split_index_1 + 1:split_index_2, :, :]
        test_data = data[split_index_2 + 1:, :, :]
    else:
        train_data = data[0:split_index_1, :]
        val_data = data[split_index_1 + 1:split_index_2, :]
        test_data = data[split_index_2 + 1:, :]

    return train_data, val_data, test_data
