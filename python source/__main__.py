"""Project main script"""

import os.path

from tensorflow.keras.callbacks import EarlyStopping
from functionality.augmentation import *
from functionality.subscription import *

# Constants
train_file = DATA_ROOT + "training.csv"


def train_model():
    """Runs the whole process of the model training."""

    # Prepare dataset if it does not exist yet
    if not os.path.isfile(DATA_ROOT + "x_train.npy"):
        preprocess_data(train_file)

    # Load the data
    x_train, x_val, x_test, y_train, y_val, y_test = load_prepared_data(DATA_ROOT)

    # Define maximum number of epochs
    epochs = 2

    # Load the model
    model, model_name = scale_cnn(0.2)

    # Hack - data size must be modulo batch size = 0
    # TODO-improvement: implement sophisticated solution
    x_train = x_train[:-(x_train.shape[0] % BATCH_SIZE), :, :]
    y_train = y_train[:-(y_train.shape[0] % BATCH_SIZE), :]
    x_val = x_val[:-(x_val.shape[0] % BATCH_SIZE), :, :]
    y_val = y_val[:-(y_val.shape[0] % BATCH_SIZE), :]

    # Create dataset from numpy arrays
    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    # Add augmentation mapping
    ds_train = ds_train.map(augment_data)
    ds_val = ds_val.map(augment_data)

    # Initialize dataset - set batch size and shuffle
    ds_train = ds_train.shuffle(buffer_size=x_train.shape[0], reshuffle_each_iteration=True).batch(BATCH_SIZE)
    ds_val = ds_val.batch(BATCH_SIZE)

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0.01,
                                   patience=10,
                                   restore_best_weights=False)

    # Train the model
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=[early_stopping]
    )

    # Save model
    # - model file name consist of the final validation loss and model type name
    model.save(MODELS_ROOT + str(history.history['val_loss'][-1]) + '_' + model_name + '.h5')

    # Show training progress in time
    plot_train_history(history)


def create_subscription():
    """Loads model with best val_loss score.
       Computes predictions.
       Creates subscription file."""

    model_name = os.listdir(MODELS_ROOT)[0]

    subscribe(model_name)


def show_presentation_data():
    """Shows some of the figures from the presentation"""

    if not os.path.isfile(DATA_ROOT + "x_train.npy"):
        preprocess_data(train_file)

    # Data
    x_train, x_val, x_test, y_train, y_val, y_test = load_prepared_data(DATA_ROOT)
    ds_train = tf.data.Dataset.from_tensor_slices((x_train[:5000], y_train[:5000]))
    ds_train = ds_train.shuffle(buffer_size=x_train.shape[0], reshuffle_each_iteration=True)
    show_images(ds_train)
    plt.show()

    # Corrupted images
    x_corrupted = np.load(DATA_ROOT + 'x_corrupted.npy').astype('float32')
    y_corrupted = np.load(DATA_ROOT + 'y_corrupted.npy', allow_pickle=True).astype('float32')
    ds_train = tf.data.Dataset.from_tensor_slices((x_corrupted, y_corrupted))
    ds_train = ds_train.shuffle(buffer_size=x_corrupted.shape[0], reshuffle_each_iteration=False)
    show_corrupted_images(ds_train)
    plt.show()


if __name__ == '__main__':
    train_model()
    # create_subscription()
    # show_presentation_data()
