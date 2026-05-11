from sklearn.preprocessing import MinMaxScaler


def scale_features(feature_data):

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(
        feature_data
    )

    return scaled_data