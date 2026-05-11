import numpy as np


def create_sequences(features, labels, sequence_length=5):

    X = []

    y = []

    for i in range(len(features) - sequence_length):

        X.append(
            features[i:i + sequence_length]
        )

        y.append(
            labels[i + sequence_length]
        )

    return np.array(X), np.array(y)