import os.path

from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from functionality.data_analysis import *
from functionality.data_operations import *
from functionality.models import *



TRAIN_FILE = DATA_ROOT + "training.csv"
TEST_FILE = DATA_ROOT + "test.csv"


def visualize_data():
    x_train, x_val, x_test, y_train, y_val, y_test = load_prepared_data(OUTPUT_ROOT)
    inspect_data(x_train, y_train)


def train_model():
    if not os.path.isfile(OUTPUT_ROOT + "x_train.npy"):
        preprocess_data(TRAIN_FILE)

    x_train, x_val, x_test, y_train, y_val, y_test = load_prepared_data(OUTPUT_ROOT)

    epochs = 500

    model = scale_cnn(0.2)

    # Hack
    x_val = x_val[:-(x_val.shape[0] % BATCH_SIZE), :, :]
    y_val = y_val[:-(y_val.shape[0] % BATCH_SIZE), :]


    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    ds_train = ds_train.shuffle(buffer_size=x_train.shape[0], reshuffle_each_iteration=True).batch(BATCH_SIZE)
    ds_val = ds_val.batch(BATCH_SIZE)

    early_stopping = EarlyStopping(monitor='loss',
                                   min_delta=0.01,
                                   patience=10,
                                   restore_best_weights=False)

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        # batch_size=BATCH_SIZE,
        callbacks=[early_stopping]
    )

    model_name = get_model_name(True)
    model.save(model_name['h5'])

    plot_train_history(history)


if __name__ == '__main__':
    train_model()
