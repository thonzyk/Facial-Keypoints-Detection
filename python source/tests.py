from functionality.data_operations import *
from functionality.constants import *
from functionality.models import *
from tensorflow.keras.losses import MeanSquaredError
from functionality.custom_losses import *

import os.path
import time

x = [i for i in range(-10, 10)]

x = tf.cast(x, tf.float32)

y = tf.divide(tf.add(tf.sign(x), 1.0), 2.0)

print("fe")
