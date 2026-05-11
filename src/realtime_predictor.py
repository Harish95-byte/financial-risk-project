import numpy as np


def predict_realtime_risk(
    model,
    transaction_sequence
):

    sequence = np.array(
        transaction_sequence
    )

    sequence = sequence.reshape(
        1,
        sequence.shape[0],
        sequence.shape[1]
    )

    prediction = model.predict(sequence)

    return float(prediction[0][0])