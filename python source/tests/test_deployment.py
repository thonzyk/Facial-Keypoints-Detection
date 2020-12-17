from functionality.deployment import *
import unittest


class TestDeployment(unittest.TestCase):
    """Tests all functions in module ´deployment´"""

    def test_create_notebook(increase_index):
        """Tests ´create_notebook´ function"""
        create_notebook()
