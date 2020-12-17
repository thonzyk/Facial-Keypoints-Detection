import pandas as pd
from .constants import *
from .data_operations import *
from tensorflow.keras.models import load_model
from .custom_losses import *
from .models import *
import os


def subscribe(model_name):
    ids, images = load_test_data()
    subscription_mapping = load_subscription_mapping()
    text_result = "RowId,Location\n"

    model_name = MODELS_ROOT + model_name

    models_names = os.listdir(MODELS_ROOT)

    predictions = np.zeros((len(ids), Y_LENGTH))

    for i in range(10):
        model_name = models_names[i]
        model, model_n = scale_cnn(0.2)
        model.load_weights(MODELS_ROOT + model_name)

        predictions = predictions + model.predict(images)

    predictions = predictions / 10

    for i, row in enumerate(predictions):
        for j, column in enumerate(row):
            key = str(ids[i]) + '_' + FEATURES_MAPPING_2[j]
            if key in subscription_mapping:
                row_id = subscription_mapping[key]
                text_result = ''.join((text_result, str(row_id), ',', str(predictions[i, j]), '\n'))


    with open(DATA_ROOT + "subscription.csv", 'w') as out_file:
        out_file.write(text_result)
