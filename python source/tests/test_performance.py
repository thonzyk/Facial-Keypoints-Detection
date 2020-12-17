import unittest
import os
import tensorflow as tf
import time


class TestPerformance(unittest.TestCase):
    """Tests performance of the code"""

    def test_gpu_matmul(self):
        tf.debugging.set_log_device_placement(True)
        n = 9216
        dtype = tf.float32
        with tf.device("/cpu:0"):
            matrix1 = tf.Variable(tf.ones((2, n), dtype=dtype))
            matrix2 = tf.Variable(tf.ones((n, 2), dtype=dtype))

        start = time.time()
        for i in range(20000):
            product = tf.matmul(matrix1, matrix2)
        end = time.time()
        print(end-start)

