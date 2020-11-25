import random
from .visualization import *


def inspect_data(x, y):
    while True:
        rand_indx = random.randint(0, x.shape[0])
        show_images(x[rand_indx:rand_indx + 1, :, :, :], y[rand_indx:rand_indx + 1, :])
