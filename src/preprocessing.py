from sklearn.preprocessing import LabelEncoder


def preprocess_data(df):

    # Remove missing values
    df = df.dropna()

    # Convert transaction types into numbers
    encoder = LabelEncoder()

    df["type"] = encoder.fit_transform(df["type"])

    return df