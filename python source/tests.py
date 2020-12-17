from functionality.data_operations import *
from functionality.constants import *
from functionality.models import *
from tensorflow.keras.losses import MeanSquaredError
from functionality.custom_losses import *

import os.path
import time


def transform_a(x, y):
    return x * 2, y + 1

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

a = tf.data.Dataset.from_tensor_slices((x, y))
a = a.batch(3)
# a = a.shuffle(buffer_size=6, reshuffle_each_iteration=True).batch(3)

np_iter_a = a.unbatch().as_numpy_iterator()

b = a.map(transform_a)

np_iter_b = b.unbatch().as_numpy_iterator()


print(b)
