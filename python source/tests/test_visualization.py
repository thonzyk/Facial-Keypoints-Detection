import unittest

from functionality.visualization import *


class TestVisualization(unittest.TestCase):

    def test_show_scales_graph(self, visual=False):
        if visual:
            show_scales_graph()
