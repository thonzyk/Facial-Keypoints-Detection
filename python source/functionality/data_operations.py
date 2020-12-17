import pandas as pd
import tensorflow as tf

from functionality.visualization import *


def preprocess_data(csv_file):
    """"""
    x_train, x_val, x_test, y_train, y_val, y_test = load_train_data(csv_file)

    # x_train, x_val, x_test, y_train, y_val, y_test = add_flipped_images(x_train, x_val, x_test, y_train, y_val, y_test)

    np.save(OUTPUT_ROOT + 'x_train.npy', x_train)
    np.save(OUTPUT_ROOT + 'x_val.npy', x_val)
    np.save(OUTPUT_ROOT + 'x_test.npy', x_test)
    np.save(OUTPUT_ROOT + 'y_train.npy', y_train)
    np.save(OUTPUT_ROOT + 'y_val.npy', y_val)
    np.save(OUTPUT_ROOT + 'y_test.npy', y_test)


def add_flipped_images(x_train, x_val, x_test, y_train, y_val, y_test):
    x_train_flip = np.zeros(x_train.shape)
    x_val_flip = np.zeros(x_val.shape)
    x_test_flip = np.zeros(x_test.shape)
    y_train_flip = np.zeros(y_train.shape)
    y_val_flip = np.zeros(y_val.shape)
    y_test_flip = np.zeros(y_test.shape)

    for i in range(x_train.shape[0]):
        x_train_flip[i, :, :, :], y_train_flip[i, :] = flip_image(x_train[i, :, :, :], y_train[i, :])

    for i in range(x_val.shape[0]):
        x_val_flip[i, :, :, :], y_val_flip[i, :] = flip_image(x_val[i, :, :, :], y_val[i, :])

    for i in range(x_test.shape[0]):
        x_test_flip[i, :, :, :], y_test_flip[i, :] = flip_image(x_test[i, :, :, :], y_test[i, :])

    x_train_flip = x_train_flip.astype('float32')
    x_val_flip = x_val_flip.astype('float32')
    x_test_flip = x_test_flip.astype('float32')
    y_train_flip = y_train_flip.astype('float32')
    y_val_flip = y_val_flip.astype('float32')
    y_test_flip = y_test_flip.astype('float32')

    x_train = np.concatenate((x_train, x_train_flip), axis=0)
    x_val = np.concatenate((x_val, x_val_flip), axis=0)
    x_test = np.concatenate((x_test, x_test_flip), axis=0)
    y_train = np.concatenate((y_train, y_train_flip), axis=0)
    y_val = np.concatenate((y_val, y_val_flip), axis=0)
    y_test = np.concatenate((y_test, y_test_flip), axis=0)

    return x_train, x_val, x_test, y_train, y_val, y_test


def flip_image(image, label):
    """Flips image and its label by vertical axis"""
    # Image flip
    image = tf.reverse(image, [1])

    # Label flip
    label_x_index = tf.tile(tf.constant([1, 0], dtype='float32'), [Y_LENGTH // 2])
    label_x = tf.subtract(tf.constant([IMAGE_SIZE[0]], dtype='float32'),
                          label * (2 * (tf.cast(label > 0, 'float32') - 0.5)))
    label_x = tf.multiply(label_x, label_x_index)

    label_y_index = tf.tile(tf.constant([0, 1], dtype='float32'), [Y_LENGTH // 2])
    label_y = tf.multiply(label, label_y_index)

    label = tf.add(label_x, label_y)

    label = np.array([label[LABEL_FLIP_MAPPING[i]] for i in range(Y_LENGTH)])

    return image, label


def load_prepared_data(directory):
    """"""
    x_train = np.load(directory + 'x_train.npy').astype('float32')
    x_val = np.load(directory + 'x_val.npy').astype('float32')
    x_test = np.load(directory + 'x_test.npy').astype('float32')
    y_train = np.load(directory + 'y_train.npy', allow_pickle=True).astype('float32')
    y_val = np.load(directory + 'y_val.npy', allow_pickle=True).astype('float32')
    y_test = np.load(directory + 'y_test.npy', allow_pickle=True).astype('float32')
    return x_train, x_val, x_test, y_train, y_val, y_test


def transform_images(images):
    # Parse pixel values from the string
    # - memory consuming operation (optimize in case of memory problems)
    images = np.array([pixels.split() for pixels in images])
    images = images.astype('float32')

    image_size = int(round(math.sqrt(images.shape[-1])))

    images = images.reshape((images.shape[0], image_size, image_size))
    images = np.divide(images, 255)
    images = np.expand_dims(images, axis=-1)

    return images


def load_train_data(csv_file):
    """"""
    data = pd.read_csv(csv_file)
    data = data.to_numpy()

    x = data[:, -1]
    y = data[:, :-1]

    # parse images from strings
    x = transform_images(x)

    # replace nan
    y = np.nan_to_num(y, nan=0)
    y[y != y] = -100000.0

    # save corrupted data aside
    corrupted_indexes = [1907, 1877, 2199, 6492, 4263, 6491, 2194, 6493]
    x_corrupted = x[corrupted_indexes, :, :, :]
    y_corrupted = y[corrupted_indexes, :]
    np.save(OUTPUT_ROOT + 'x_corrupted.npy', x_corrupted)
    np.save(OUTPUT_ROOT + 'y_corrupted.npy', y_corrupted)

    # remove corrupted images
    corrupted_indexes = np.array(corrupted_indexes)
    indexes = [i for i in range(x.shape[0]) if i not in corrupted_indexes]
    x = x[indexes, :, :, :]
    y = y[indexes, :]

    # Shuffle both x and y by the same permutation
    shuffle_permutation = np.random.permutation(x.shape[0])
    x = x[shuffle_permutation]
    y = y[shuffle_permutation]

    x_train, x_val, x_test = split_data(x)
    y_train, y_val, y_test = split_data(y)

    return x_train.astype('float32'), x_val.astype('float32'), x_test.astype('float32'), y_train.astype(
        'float32'), y_val.astype('float32'), y_test.astype('float32')


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

    # TODO-refactor: more sophisticated solution
    if len(data.shape) == 3:
        train_data = data[0:split_index_1, :, :]
        val_data = data[split_index_1 + 1:split_index_2, :, :]
        test_data = data[split_index_2 + 1:, :, :]
    else:
        train_data = data[0:split_index_1, :]
        val_data = data[split_index_1 + 1:split_index_2, :]
        test_data = data[split_index_2 + 1:, :]

    return train_data, val_data, test_data
