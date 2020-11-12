from functionality.data_operations import *
import unittest


class TestDataOperations(unittest.TestCase):
    """Tests all functions in module subscription"""

    def test_load_test_data(self):
        """Tests load_test_data function"""
        load_test_data()

    def test_load_subscription_mapping(self):
        """Tests load_test_data function"""
        load_subscription_mapping()
