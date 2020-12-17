import os.path

from tensorflow.keras.callbacks import EarlyStopping

from functionality.data_analysis import *
from functionality.data_operations import *
from functionality.models import *
from functionality.augmentation import *

TRAIN_FILE = DATA_ROOT + "training.csv"
TEST_FILE = DATA_ROOT + "test.csv"


def visualize_data():
    x_train, x_val, x_test, y_train, y_val, y_test = load_prepared_data(OUTPUT_ROOT)
    inspect_data(x_train, y_train)


def train_model():
    if not os.path.isfile(OUTPUT_ROOT + "x_train.npy"):
        preprocess_data(TRAIN_FILE)

    x_train, x_val, x_test, y_train, y_val, y_test = load_prepared_data(OUTPUT_ROOT)

    epochs = 1

    model, model_name = scale_cnn(0.2)
    # model = scale_1()

    # Hack
    x_train = x_train[:-(x_train.shape[0] % BATCH_SIZE), :, :]
    y_train = y_train[:-(y_train.shape[0] % BATCH_SIZE), :]
    x_val = x_val[:-(x_val.shape[0] % BATCH_SIZE), :, :]
    y_val = y_val[:-(y_val.shape[0] % BATCH_SIZE), :]

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    ds_train = ds_train.map(augment_data)

    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    ds_val = ds_val.map(augment_data)

    ds_train = ds_train.shuffle(buffer_size=x_train.shape[0], reshuffle_each_iteration=True).batch(BATCH_SIZE)
    ds_val = ds_val.batch(BATCH_SIZE)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0.01,
                                   patience=10,
                                   restore_best_weights=False)

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=[early_stopping]
    )

    # model_name = get_model_name(True)
    model.save(MODELS_ROOT + str(history.history['val_loss'][-1]) + '_' + model_name + '.h5')

    plot_train_history(history)


def show_presentation_data():
    if not os.path.isfile(OUTPUT_ROOT + "x_train.npy"):
        preprocess_data(TRAIN_FILE)

    # # Data
    # x_train, x_val, x_test, y_train, y_val, y_test = load_prepared_data(OUTPUT_ROOT)
    # ds_train = tf.data.Dataset.from_tensor_slices((x_train[:5000], y_train[:5000]))
    # ds_train = ds_train.shuffle(buffer_size=x_train.shape[0], reshuffle_each_iteration=True)
    # show_images(ds_train)
    # plt.show()

    # Corrupted images
    x_corrupted = np.load(DATA_ROOT + 'x_corrupted.npy').astype('float32')
    y_corrupted = np.load(DATA_ROOT + 'y_corrupted.npy', allow_pickle=True).astype('float32')
    ds_train = tf.data.Dataset.from_tensor_slices((x_corrupted, y_corrupted))
    ds_train = ds_train.shuffle(buffer_size=x_corrupted.shape[0], reshuffle_each_iteration=False)
    show_corrupted_images(ds_train)
    plt.show()



if __name__ == '__main__':
    train_model()
    # show_presentation_data()
