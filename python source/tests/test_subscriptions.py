import os
import unittest

from functionality.subscription import *


class TestSubscriptions(unittest.TestCase):
    """Tests all functions in module subscription"""

    def test_subscribe(self):
        """Tests the ´subscribe´ function"""

        model_name = os.listdir(MODELS_ROOT)[0]

        subscribe(model_name)
