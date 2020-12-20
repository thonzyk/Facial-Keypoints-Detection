"""Models"""

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input, concatenate
from tensorflow.keras.models import Model

from .custom_losses import *


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

    return model, 'scale_cnn_1_4'
