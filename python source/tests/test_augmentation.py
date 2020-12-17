import unittest
import numpy as np

from functionality.augmentation import *
from functionality.data_operations import *
from functionality.visualization import *


class TestAugmentation(unittest.TestCase):

    def test_get_augment_matrix(self):
        get_augment_matrix(10.0, 1.0, -2.0)

    def test_augment_data(self, visual=True):
        x_train, x_val, x_test, y_train, y_val, y_test = load_prepared_data(OUTPUT_ROOT)

        augment_data(x_train[1, :, :], y_train[1, :])

        ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        ds_train_aug = ds_train.map(augment_data)

        rand_seed = np.random.randint(-1000000, 1000000)
        ds_train_aug = ds_train_aug.shuffle(buffer_size=x_train.shape[0], reshuffle_each_iteration=True, seed=rand_seed)
        ds_train = ds_train.shuffle(buffer_size=x_train.shape[0], reshuffle_each_iteration=True, seed=rand_seed)

        show_images(ds_train)
        show_images(ds_train_aug)

        plt.show()
