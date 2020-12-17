import tensorflow as tf
import numpy as np
import random

from functionality.constants import *

deg_180 = tf.constant([180], dtype='float32')
_1 = tf.constant([1], dtype='float32')
_0 = tf.constant([0], dtype='float32')
mat_shape = (3, 3)


def get_augment_matrix(rot=0.0, shift_x=0.0, shift_y=0.0, zoom_x=0.0, zoom_y=0.0):
    """"""
    # Degrees to rad
    rot = np.pi * (rot / deg_180)

    # Rotation matrix
    sin = tf.math.sin(rot)
    cos = tf.math.cos(rot)
    rot_mat = tf.concat([cos, sin, _0,
                         -sin, cos, _0,
                         _0, _0, _1], axis=0)
    rot_mat = tf.reshape(rot_mat, mat_shape)

    # Shift matrix
    shift_mat = tf.concat([_1, _0, shift_y,
                           _0, _1, shift_x,
                           _0, _0, _1], axis=0)
    shift_mat = tf.reshape(shift_mat, mat_shape)

    # Zoom matrix
    zoom_mat = tf.concat([_1 / (_1 + zoom_y), _0, _0,
                          _0, _1 / (_1 + zoom_x), _0,
                          _0, _0, _0], axis=0)
    zoom_mat = tf.reshape(zoom_mat, mat_shape)
    ###################################################
    # Rotation matrix
    rot_mat_2 = tf.concat([cos, -sin, _0,
                           sin, cos, _0,
                           _0, _0, _1], axis=0)
    rot_mat_2 = tf.reshape(rot_mat_2, mat_shape)

    # Shift matrix
    shift_mat_2 = tf.concat([_1, _0, -shift_x,
                             _0, _1, shift_y,
                             _0, _0, _1], axis=0)
    shift_mat_2 = tf.reshape(shift_mat_2, mat_shape)

    # Zoom matrix
    zoom_mat_2 = tf.concat([_1 + zoom_x, _0, _0,
                            _0, _1 + zoom_y, _0,
                            _0, _0, _0], axis=0)
    zoom_mat_2 = tf.reshape(zoom_mat_2, mat_shape)
    ###################################################

    # Combine matrices by simple multiplication
    return tf.matmul(tf.matmul(rot_mat, shift_mat), zoom_mat), tf.matmul(tf.matmul(shift_mat_2, rot_mat_2), zoom_mat_2)


def augment_data(image, labels):
    """"""
    # Generate transformation parameters
    rot = tf.random.uniform(shape=(1,), minval=-15, maxval=15, dtype='float32')
    shift_x = tf.random.uniform(shape=(1,), minval=-15, maxval=15, dtype='float32')
    shift_y = tf.random.uniform(shape=(1,), minval=-15, maxval=15, dtype='float32')
    zoom = tf.random.normal([1], 0, 0.15, dtype='float32')

    augment_mat, augment_mat_2 = get_augment_matrix(rot, shift_x, shift_y, zoom, zoom)

    # Image pixels coordinates
    # - x and y vectors: so their stack makes cartesian product of pixel coordinates (all possible coordinate pair)
    # - intercept vector: serves as supportive vector for shift operation (allows the operation in absence of addition)
    x = tf.repeat(tf.range(IMAGE_SIZE[0] // 2, -IMAGE_SIZE[0] // 2, -1), IMAGE_SIZE[0])
    y = tf.tile(tf.range(-IMAGE_SIZE[1] // 2, IMAGE_SIZE[1] // 2), [IMAGE_SIZE[1]])
    intercept = tf.ones([IMAGE_SIZE[0] * IMAGE_SIZE[1]], dtype='int32')
    coordinates = tf.stack([x, y, intercept])

    # Find new pixel coordinates
    # 1. Multiply coordinates and augmentation matrix
    # 2. Remove coordinates exceeding the space of the image
    # 3. Get rid of supportive vector and update the reference frame
    coordinates = tf.cast(tf.matmul(augment_mat, tf.cast(coordinates, dtype='float32')), dtype='int32')
    coordinates = tf.keras.backend.clip(coordinates, -IMAGE_SIZE[0] // 2 + 1, IMAGE_SIZE[1] // 2)
    coordinates = tf.stack([IMAGE_SIZE[0] // 2 - coordinates[0,], IMAGE_SIZE[1] // 2 - 1 + coordinates[1,]])

    # Map the image to the new reference frame
    image = tf.gather_nd(image, tf.transpose(coordinates))
    image = tf.reshape(image, [IMAGE_SIZE[0], IMAGE_SIZE[1], 1])

    # Labels to x and y coordinate
    # labels = tf.boolean_mask(labels, tf.math.greater(labels, 0))
    x = tf.concat(labels[0::2], axis=0) - (IMAGE_SIZE[0] / 2)
    y = tf.concat(labels[1::2], axis=0) - (IMAGE_SIZE[1] / 2)
    intercept = tf.ones([tf.size(x)], dtype='float32')
    coordinates = tf.stack([x, y, intercept])

    coordinates = tf.matmul(augment_mat_2, tf.cast(coordinates, dtype='float32'))
    coordinates = tf.stack([coordinates[0,] + IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2 + coordinates[1,]])

    labels = tf.reshape(tf.stack([coordinates[0], coordinates[1]], axis=1), [tf.size(x) * 2])

    return image, labels
