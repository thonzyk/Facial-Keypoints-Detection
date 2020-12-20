"""Subscription"""

from .data_operations import *
from .models import *


def subscribe(model_name):
    """Loads the specified model.
       Computes the model predictions.
       Creates the subscription file."""

    ids, images = load_test_data()
    subscription_mapping = load_subscription_mapping()
    text_result = "RowId,Location\n"

    predictions = np.zeros((len(ids), Y_LENGTH))

    model, model_n = scale_cnn(0.2)
    model.load_weights(MODELS_ROOT + model_name)

    predictions = predictions + model.predict(images)

    for i, row in enumerate(predictions):
        for j, column in enumerate(row):
            key = str(ids[i]) + '_' + FEATURES_MAPPING_2[j]
            if key in subscription_mapping:
                row_id = subscription_mapping[key]
                text_result = ''.join((text_result, str(row_id), ',', str(predictions[i, j]), '\n'))

    with open(DATA_ROOT + "subscription.csv", 'w') as out_file:
        out_file.write(text_result)
