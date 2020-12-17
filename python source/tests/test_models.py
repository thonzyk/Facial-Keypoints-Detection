from functionality.models import *
import unittest
import os


class TestModels(unittest.TestCase):
    """Tests all functions in module ´models´"""

    def test_get_model_name(increase_index):
        """Tests ´get_model_name´ function"""
        model_name = get_model_name(False)

        assert os.path.isfile(model_name['h5'])

    def test_shapes(self):
        shape_test()
