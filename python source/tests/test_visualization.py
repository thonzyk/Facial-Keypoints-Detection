import unittest
import numpy as np

from functionality.visualization import *


class TestVisualization(unittest.TestCase):

    def test_show_scales_graph(self):
        show_scales_graph()
