from functionality.data_operations import *
from functionality.constants import *
from functionality.models import *
from tensorflow.keras.losses import MeanSquaredError
from functionality.custom_losses import *

import os.path
import time

TRAIN_FILE = DATA_ROOT + "training.csv"
TEST_FILE = DATA_ROOT + "test.csv"

if __name__ == '__main__':
    if not os.path.isfile(DATA_ROOT + "x_train.npy"):
        preprocess_data(TRAIN_FILE)

    x_train, x_val, x_test, y_train, y_val, y_test = load_prepared_data(DATA_ROOT)

    EPOCHS = 10

    model = shallow_convolution()

    model.compile(
        optimizer='adam',
        loss=root_mse_with_exceptions,
        metrics=[],
    )

    model.summary()

    y_train = np.nan_to_num(y_train, nan=0)
    y_val = np.nan_to_num(y_val, nan=0)

    # replace nan
    y_train[y_train != y_train] = -1.0
    y_val[y_val != y_val] = -1.0

    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    x_val = x_val.astype('float32')
    y_val = y_val.astype('float32')

    # Hack
    x_val = x_val[:-(x_val.shape[0] % BATCH_SIZE), :, :]
    y_val = y_val[:-(y_val.shape[0] % BATCH_SIZE), :]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    model_name = get_model_name(True)
    model.save(model_name['h5'])

    # plt.figure()
    # plt.plot(history.history['loss'], label='RMSE')
    # plt.xlabel('Epoch')
    # plt.legend(['RMSE'])
    # plt.show()




