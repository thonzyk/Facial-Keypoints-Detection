import unittest
import os
import tensorflow as tf

from functionality.models import *
from functionality.data_operations import *
from functionality.custom_losses import *
from functionality.data_analysis import *


class TestData(unittest.TestCase):
    """Tests data consistency"""

    def test_model_index(increase_index):
        """Tests ´model_index.txt´ existence"""
        model_name = get_model_name()
        assert os.path.isfile(model_name['h5'])

    def test_data_corruption(self):
        x_train, x_val, x_test, y_train, y_val, y_test = load_prepared_data(OUTPUT_ROOT)

        x = np.concatenate((x_train, x_val, x_test), axis=0)
        y = np.concatenate((y_train, y_val, y_test), axis=0)
        model, model_n = scale_cnn(0.2)
        model.load_weights(MODELS_ROOT + "1.9005311727523804_scale_cnn_1_1.h5")

        predictions = model.predict(x)
        err = predictions - y
        err = np.where(err > 10000, 0, err)  # null excluded elements
        err = np.abs(err)
        err_sum = np.sum(err, axis=1).reshape(err.shape[0], 1)

        err = np.concatenate((err_sum, err), axis=1)

        # Sort by err sum
        df = pd.DataFrame(err,
                          columns=['Sum', 'X_1', 'Y_1', 'X_2', 'Y_2', 'X_3', 'Y_3', 'X_4', 'Y_4', 'X_5', 'Y_5', 'X_6',
                                   'Y_6', 'X_7', 'Y_7', 'X_8', 'Y_8', 'X_9', 'Y_9', 'X_10', 'Y_10', 'X_11', 'Y_11',
                                   'X_12', 'Y_12', 'X_13', 'Y_13', 'X_14', 'Y_14', 'X_15', 'Y_15'])
        df = df.sort_values(by=['Sum'], ascending=True)
        df.drop(columns=['Sum'])
        err = df.to_numpy()

        indexes = df.index[:]
        x_what = x[indexes, :, :, :]
        y_what = y[indexes, :]
        pred_what = predictions[indexes, :]

        show_images_2(x_what, y_what, pred_what, indexes)

    def test_statistics(self):
        statistics()
