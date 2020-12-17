# -*- coding: utf-8 -*-
"""Set of different models returned by each function"""

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201, VGG16, VGG19
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input, concatenate
from tensorflow.keras.models import Sequential, Model
import json
import numpy as np

from .custom_losses import *

# INITIALIZATION
# Detect TPU, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()


# MODELS
def tutorial_model():
    """Copy of the model from the tutorial notebook
    https://www.kaggle.com/ryanholbrook/create-your-first-submission
    """
    with strategy.scope():
        pretrained_model = tf.keras.applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=[*IMAGE_SIZE, 3]
        )
        pretrained_model.trainable = False

        model = tf.keras.Sequential([
            # To a base pretrained on ImageNet to extract features from images...
            pretrained_model,
            # ... attach a new head to act as a classifier.
            tf.keras.layers.GlobalAveragePooling2D(),
            Dense(Y_LENGTH, activation=None)
        ])

    return model


def pretrainded_model(type: str, trainable=False):
    with strategy.scope():
        if type == 'VGG16':
            pretrained_model = VGG16(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
        elif type == 'VGG19':
            pretrained_model = VGG19(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
        elif type == 'DenseNet121':
            pretrained_model = DenseNet121(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
        elif type == 'DenseNet169':
            pretrained_model = DenseNet169(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
        elif type == 'DenseNet201':
            pretrained_model = DenseNet201(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])

        pretrained_model.trainable = trainable

        model = Sequential([
            # To a base pretrained on ImageNet to extract features from images...
            pretrained_model,
            # ... attach a new head to act as a classifier.
            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(Y_LENGTH, activation=None)
        ])

    return model


def vgg16(trainable=False):
    return pretrainded_model('VGG16', trainable)


def vgg19(trainable=False):
    return pretrainded_model('VGG19', trainable)


def densenet121(trainable=False):
    return pretrainded_model('DenseNet121', trainable)


def densenet169(trainable=False):
    return pretrainded_model('DenseNet169', trainable)


def densenet201(trainable=False):
    return pretrainded_model('DenseNet201', trainable)


def deep_convolution(overfit=False):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        # Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        # Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        # Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        # Dropout(0.2),
        Flatten(),
        Dense(256, activation='relu'),
        # Dropout(0.2),
        Dense(256, activation='relu'),
        # Dropout(0.2),
        Dense(256, activation='relu'),
        # Dropout(0.2),
        Dense(256, activation='relu'),
        # Dropout(0.2),
        Dense(256, activation='relu'),
        # Dropout(0.2),
        Dense(Y_LENGTH, activation=None)
    ])

    return model


def shallow_convolution(overfit=False):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Dropout(0.0 if overfit else 0.2),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Dropout(0.0 if overfit else 0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.0 if overfit else 0.2),
        Dense(512, activation='relu'),
        Dropout(0.0 if overfit else 0.2),
        Dense(512, activation='relu'),
        Dropout(0.0 if overfit else 0.2),
        Dense(512, activation='relu'),
        Dropout(0.0 if overfit else 0.2),
        Dense(512, activation='relu'),
        Dropout(0.0 if overfit else 0.2),
        Dense(512, activation='relu'),
        Dropout(0.0 if overfit else 0.2),
        Dense(512, activation='relu'),
        Dropout(0.0 if overfit else 0.2),
        Dense(512, activation='relu'),
        Dropout(0.0 if overfit else 0.2),
        Dense(Y_LENGTH, activation=None)
    ])

    model.compile(
        optimizer='adam',
        loss=root_mse_with_exceptions,
        metrics=[]
    )

    model.summary()

    return model


def scale_cnn(dropout):
    input_layer = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1))

    # 1st scale
    s01_l01 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(input_layer)
    s01_l02 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(s01_l01)
    s01_l03 = MaxPool2D((2, 2))(s01_l02)
    s01_l04 = Dropout(dropout)(s01_l03)
    s01_output = Flatten()(s01_l04)

    # 2nd scale
    s02_l01 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(s01_l04)
    s02_l02 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(s02_l01)
    s02_l03 = MaxPool2D((2, 2))(s02_l02)
    s02_l04 = Dropout(dropout)(s02_l03)
    s02_output = Flatten()(s02_l04)

    # 3rd scale
    s03_l01 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu')(s02_l04)
    s03_l02 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu')(s03_l01)
    s03_l03 = MaxPool2D((2, 2))(s03_l02)
    s03_l04 = Dropout(dropout)(s03_l03)
    s03_l05 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu')(s03_l04)
    s03_l06 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu')(s03_l05)
    s03_l07 = MaxPool2D((2, 2))(s03_l06)
    s03_l08 = Dropout(dropout)(s03_l07)
    s03_output = Flatten()(s03_l08)

    # Merge all network scales
    merge_layer = concatenate([s01_output, s02_output, s03_output])
    output_layer = Dense(Y_LENGTH, activation=None, use_bias=False)(merge_layer)
    model = Model(inputs=[input_layer], outputs=[output_layer])




    model.compile(
        optimizer='adam',
        loss=root_mse_with_exceptions,
        metrics=[],
    )

    model.summary()

    return model, 'scale_cnn_1_1'


def scale_cnn_3(dropout):
    """
    Notation example:
    ´s01_l03´ means the 3rd hidden layer of the 1st scale network
    """

    input_layer = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1))

    # 1st scale
    s01_l01 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu')(input_layer)
    s01_l02 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu')(s01_l01)
    s01_l03 = MaxPool2D((2, 2))(s01_l02)
    s01_l04 = Dropout(dropout)(s01_l03)
    s01_output = Flatten()(s01_l04)

    # 2nd scale
    s02_l01 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(s01_l04)
    s02_l02 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(s02_l01)
    s02_l03 = MaxPool2D((2, 2))(s02_l02)
    s02_l04 = Dropout(dropout)(s02_l03)
    s02_output = Flatten()(s02_l04)

    # 3rd scale
    s03_l01 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu')(s02_l04)
    s03_l02 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu')(s03_l01)
    s03_l03 = MaxPool2D((2, 2))(s03_l02)
    s03_l04 = Dropout(dropout)(s03_l03)
    s03_l05 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu')(s03_l04)
    s03_l06 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu')(s03_l05)
    s03_l07 = MaxPool2D((2, 2))(s03_l06)
    s03_l08 = Dropout(dropout)(s03_l07)
    s03_output = Flatten()(s03_l08)

    # Merge all network scales
    merge_layer = concatenate([s01_output, s02_output, s03_output])

    output_layer = Dense(Y_LENGTH, activation=None, use_bias=False)(merge_layer)

    model = Model(inputs=[input_layer], outputs=[output_layer])

    model.compile(
        optimizer='adam',
        loss=root_mse_with_exceptions,
        metrics=[],
    )

    model.summary()

    return model, 'scale_cnn_3_4'


def empty_mode():
    model = Sequential([
        Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
        Flatten(),
        Dense(Y_LENGTH, activation=None)
    ])

    model.compile(
        optimizer='adam',
        loss=root_mse_with_exceptions,
        metrics=[]
    )

    model.summary()

    return model


def scale_1():
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
        Flatten(),
        Dense(Y_LENGTH, activation=None)
    ])

    model.compile(
        optimizer='adam',
        loss=root_mse_with_exceptions,
        metrics=[]
    )

    model.summary()

    return model


def get_model_name(increase_index=False):
    digits = -1
    fname = PROGRAM_DATA_ROOT + "model_index.txt"
    with open(fname) as f:
        index = f.readlines()
        digits = len(index[0])

    if increase_index:
        index = str(int(index[0]) + 1)
        index = (digits - len(index)) * '0' + index
        with open(fname, "w") as f:
            f.write(index)
    else:
        index = index[0]

    model_name = MODELS_ROOT + index + "_Facial_Keypoints_Detection"

    result = dict()
    result['h5'] = model_name + ".h5"
    result['json'] = model_name + ".json"

    return result


def shape_test():
    model = Sequential([
        Conv2D(8, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
        Conv2D(8, (3, 3), activation='relu'),
        Conv2D(8, (3, 3), activation='relu'),
        Conv2D(8, (3, 3), activation='relu'),
        Conv2D(8, (3, 3), activation='relu'),
        Conv2D(8, (3, 3), activation='relu'),
        Conv2D(8, (3, 3), activation='relu'),
    ])

    model.compile(
        optimizer='adam',
        loss=root_mse_with_exceptions,
        metrics=[]
    )

    model.summary()

    pred = model.predict(np.zeros((1, 96, 96, 1)))

    return model
