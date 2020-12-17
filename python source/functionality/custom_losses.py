"""Custom loss functions"""

import tensorflow as tf
from .constants import *

y_size = tf.cast(Y_LENGTH, tf.float32)


## AUXILIARY FUNCTIONS

def positive_sign(x):
    return tf.divide(tf.add(tf.sign(x), 1.0), 2.0)


## LOSSES

def custom_mse(y_true, y_predict):
    diff = tf.subtract(y_true, y_predict)
    return tf.reduce_mean(tf.pow(diff, 2.0), axis=-1)


def root_mse_with_exceptions(y_true, y_predict, batch_siz=tf.constant(BATCH_SIZE, tf.float32)):
    """Computes root mean squared error (RMSE) for positive values of `y_true`.
    Negative values of `y_true` represent exceptions and thus have no contribution to the loss function."""

    # Warning: Length of the dataset modulo `BATCH_SIZE` must be equal to 0 TODO-improvement: generalize

    # 1) Exclude exceptions via mask
    exception_mask = positive_sign(y_true)
    y_true = tf.multiply(y_true, exception_mask)
    y_predict = tf.multiply(y_predict, exception_mask)

    # 2) Compute divisor with respect to number of excluded items
    divisor = tf.reduce_sum(exception_mask, axis=-1)  # Potential problem: could be equal to 0 TODO-improvement: solve
    divisor_mat = tf.divide(tf.eye(batch_siz), divisor)

    # 3) Compute MSE with respect to the divisor value
    diff = tf.subtract(y_true, y_predict)
    mse = tf.tensordot(divisor_mat, tf.reduce_sum(tf.pow(diff, 2.0), axis=-1), axes=1)

    # 4) Return square-root of custom MSE loss function
    return tf.pow(mse, 0.5)
