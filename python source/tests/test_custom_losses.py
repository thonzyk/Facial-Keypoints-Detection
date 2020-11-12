from functionality.custom_losses import *
import unittest
import numpy as np

MAX_ERR = pow(10.0, -6.0)


class TestCustomLosses(unittest.TestCase):
    """Tests all functions in module custom_losses"""

    def test_positive_sign(self):
        """Tests positive_sign function"""
        x = np.arange(-1.0, 1.0, 0.1)
        y = positive_sign(x)

        for i in range(len(y)):
            if x[i] < 0:
                self.assertTrue(abs(y[i]) < MAX_ERR)
            elif x[i] > 0:
                self.assertTrue(abs(y[i] - 1.0) < MAX_ERR)

    def test_root_mse_with_exceptions(self):
        """Tests root_mse_with_exceptions function"""

        batch_size = 4

        a = np.array([[1.0, -1.0, 3.0],
                      [2.0, 3.0, 4.0],
                      [3.0, 4.0, -1.0],
                      [-1.0, 5.0, 6.0]])
        b = np.ones((batch_size, 3))

        a = a.astype('float32')
        b = b.astype('float32')

        loss = root_mse_with_exceptions(a, b, 4)
        loss_expected = np.array([np.sqrt(4.0/2.0), np.sqrt(14.0/3.0), np.sqrt(13.0/2.0), np.sqrt(41.0/2.0)])
        loss_diff = tf.subtract(loss, loss_expected)

        for diff in loss_diff:
            self.assertTrue(abs(diff) < MAX_ERR)
