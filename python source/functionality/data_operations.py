"""Data operations"""

import math

import pandas as pd
import tensorflow as tf

from functionality.visualization import *


def preprocess_data(csv_file):
    """Preprocess data from csv file and saves it as numpy array in .npy format."""

    # Load data
    x_train, x_val, x_test, y_train, y_val, y_test = load_train_data(csv_file)

    # Add mirror flip augmentation
    x_train, x_val, x_test, y_train, y_val, y_test = add_flipped_images(x_train, x_val, x_test, y_train, y_val, y_test)

    # Save as .npy
    np.save(DATA_ROOT + 'x_train.npy', x_train)
    np.save(DATA_ROOT + 'x_val.npy', x_val)
    np.save(DATA_ROOT + 'x_test.npy', x_test)
    np.save(DATA_ROOT + 'y_train.npy', y_train)
    np.save(DATA_ROOT + 'y_val.npy', y_val)
    np.save(DATA_ROOT + 'y_test.npy', y_test)


def add_flipped_images(x_train, x_val, x_test, y_train, y_val, y_test):
    """Adds mirror flip augmentation to the input data"""

    # Declare augmented data variables
    x_train_flip = np.zeros(x_train.shape)
    x_val_flip = np.zeros(x_val.shape)
    x_test_flip = np.zeros(x_test.shape)
    y_train_flip = np.zeros(y_train.shape)
    y_val_flip = np.zeros(y_val.shape)
    y_test_flip = np.zeros(y_test.shape)

    # Flip images and labels
    # TODO-improvement: implement for-free solution
    # TODO-improvement: remove code redundancy
    for i in range(x_train.shape[0]):
        x_train_flip[i, :, :, :], y_train_flip[i, :] = flip_image(x_train[i, :, :, :], y_train[i, :])

    for i in range(x_val.shape[0]):
        x_val_flip[i, :, :, :], y_val_flip[i, :] = flip_image(x_val[i, :, :, :], y_val[i, :])

    for i in range(x_test.shape[0]):
        x_test_flip[i, :, :, :], y_test_flip[i, :] = flip_image(x_test[i, :, :, :], y_test[i, :])

    # Cast to float32
    x_train_flip = x_train_flip.astype('float32')
    x_val_flip = x_val_flip.astype('float32')
    x_test_flip = x_test_flip.astype('float32')
    y_train_flip = y_train_flip.astype('float32')
    y_val_flip = y_val_flip.astype('float32')
    y_test_flip = y_test_flip.astype('float32')

    # Concatenate original and flipped data
    x_train = np.concatenate((x_train, x_train_flip), axis=0)
    x_val = np.concatenate((x_val, x_val_flip), axis=0)
    x_test = np.concatenate((x_test, x_test_flip), axis=0)
    y_train = np.concatenate((y_train, y_train_flip), axis=0)
    y_val = np.concatenate((y_val, y_val_flip), axis=0)
    y_test = np.concatenate((y_test, y_test_flip), axis=0)

    return x_train, x_val, x_test, y_train, y_val, y_test


def flip_image(image, label):
    """Flips image and its label by vertical axis."""

    # Image flip
    image = tf.reverse(image, [1])

    # Declare X labels filter
    label_x_index = tf.tile(tf.constant([1, 0], dtype='float32'), [Y_LENGTH // 2])

    # Flip labels coordinates
    label_x = tf.subtract(tf.constant([IMAGE_SIZE[0]], dtype='float32'),
                          label * (2 * (tf.cast(label > 0, 'float32') - 0.5)))

    # Filter X labels
    label_x = tf.multiply(label_x, label_x_index)

    # Declare Y labels filter
    label_y_index = tf.tile(tf.constant([0, 1], dtype='float32'), [Y_LENGTH // 2])

    # Filter Y labels
    label_y = tf.multiply(label, label_y_index)

    # Merge X and Y labels
    label = tf.add(label_x, label_y)

    # Map mirror labels
    label = np.array([label[LABEL_FLIP_MAPPING[i]] for i in range(Y_LENGTH)])

    return image, label


def load_prepared_data(directory):
    """Loads prepared data."""

    x_train = np.load(directory + 'x_train.npy').astype('float32')
    x_val = np.load(directory + 'x_val.npy').astype('float32')
    x_test = np.load(directory + 'x_test.npy').astype('float32')
    y_train = np.load(directory + 'y_train.npy', allow_pickle=True).astype('float32')
    y_val = np.load(directory + 'y_val.npy', allow_pickle=True).astype('float32')
    y_test = np.load(directory + 'y_test.npy', allow_pickle=True).astype('float32')

    return x_train, x_val, x_test, y_train, y_val, y_test


def transform_images(images):
    """Takes images in string format (space-separated pixel values 0-255)
       and returns the images as numpy arrays (with pixel values 0.0-1.0)."""

    # Parse pixel values from the string
    # TODO-improvement: memory-consuming and slow operation - optimize
    images = np.array([pixels.split() for pixels in images])
    images = images.astype('float32')

    # Compute image size assuming square shape of the images
    image_size = int(round(math.sqrt(images.shape[-1])))

    # Images format
    images = images.reshape((images.shape[0], image_size, image_size))
    images = np.divide(images, 255)
    images = np.expand_dims(images, axis=-1)

    return images


def load_train_data(csv_file):
    """Loads the training data from csv file."""

    # Load data
    data = pd.read_csv(csv_file)
    data = data.to_numpy()

    # Split images ´x´ and labels ´y´
    x = data[:, -1]
    y = data[:, :-1]

    # Parse images list of strings
    x = transform_images(x)

    # Replace nan with big negative value
    # - necessary for the custom loss function (specifically for the sign(x) function)
    y = np.nan_to_num(y, nan=0)
    y[y != y] = -100000.0

    # Save corrupted data aside (indexes found manually)
    corrupted_indexes = [1907, 1877, 2199, 6492, 4263, 6491, 2194, 6493]
    x_corrupted = x[corrupted_indexes, :, :, :]
    y_corrupted = y[corrupted_indexes, :]
    np.save(DATA_ROOT + 'x_corrupted.npy', x_corrupted)
    np.save(DATA_ROOT + 'y_corrupted.npy', y_corrupted)

    # Remove corrupted images
    corrupted_indexes = np.array(corrupted_indexes)
    indexes = [i for i in range(x.shape[0]) if i not in corrupted_indexes]
    x = x[indexes, :, :, :]
    y = y[indexes, :]

    # Shuffle both x and y by the same permutation
    shuffle_permutation = np.random.permutation(x.shape[0])
    x = x[shuffle_permutation]
    y = y[shuffle_permutation]

    # Split the dataset into train-val-test parts.
    x_train, x_val, x_test = split_data(x)
    y_train, y_val, y_test = split_data(y)

    return x_train.astype('float32'), x_val.astype('float32'), x_test.astype('float32'), y_train.astype(
        'float32'), y_val.astype('float32'), y_test.astype('float32')


def load_test_data(csv_file="test.csv"):
    """Loads the test data."""
    data = pd.read_csv(DATA_ROOT + csv_file)
    data = data.to_numpy()

    ids = data[:, 0]
    images = data[:, 1]

    images = transform_images(images)

    return ids, images


def load_subscription_mapping(csv_file="IdLookupTable.csv"):
    """Loads the subscription mapping."""

    data = pd.read_csv(DATA_ROOT + csv_file)
    mapping = dict()

    for index, row in data.iterrows():
        key = str(row['ImageId']) + '_' + row['FeatureName']
        mapping[key] = int(row["RowId"])

    return mapping


def split_data(data, train_portion=0.9, val_portion=None):
    """Splits data into train-val-test parts."""
    # TODO-refactor: more sophisticated solution

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

    if len(data.shape) == 3:
        train_data = data[0:split_index_1, :, :]
        val_data = data[split_index_1 + 1:split_index_2, :, :]
        test_data = data[split_index_2 + 1:, :, :]
    else:
        train_data = data[0:split_index_1, :]
        val_data = data[split_index_1 + 1:split_index_2, :]
        test_data = data[split_index_2 + 1:, :]

    return train_data, val_data, test_data
