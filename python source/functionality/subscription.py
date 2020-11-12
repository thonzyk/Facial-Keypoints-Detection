import pandas as pd
from .constants import *
from .data_operations import *
from tensorflow.keras.models import load_model
from .custom_losses import *


def subscribe(model_name=None):
    ids, images = load_test_data()
    subscription_mapping = load_subscription_mapping()
    text_result = "RowId,Location\n"

    if not model_name:
        model_name = get_model_name(False)

    model = load_model(model_name['h5'], custom_objects={'root_mse_with_exceptions': root_mse_with_exceptions})

    predictions = model.predict(images)

    for i, row in enumerate(predictions):
        for j, column in enumerate(row):
            key = str(ids[i]) + '_' + FEATURES_MAPPING_2[j]
            if key in subscription_mapping:
                row_id = subscription_mapping[key]
                text_result = ''.join((text_result, str(row_id), ',', str(predictions[i, j]), '\n'))

    with open(DATA_ROOT + "subscription.csv", 'w') as out_file:
        out_file.write(text_result)
