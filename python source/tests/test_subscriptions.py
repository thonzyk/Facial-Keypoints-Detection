from functionality.subscription import *
import unittest


class TestSubscriptions(unittest.TestCase):
    """Tests all functions in module subscription"""

    def test_subscribe(self):
        """Tests the ´subscribe´ function"""
        subscribe('1.9005311727523804_scale_cnn_1_1.h5')