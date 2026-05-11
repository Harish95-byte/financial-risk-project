from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def build_lstm_model(input_shape):

    model = Sequential()

    model.add(
        LSTM(
            64,
            input_shape=input_shape
        )
    )

    model.add(
        Dense(1, activation="sigmoid")
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
def train_lstm_model(model, X_seq, y_seq):

    history = model.fit(
        X_seq,
        y_seq,
        epochs=3,
        batch_size=32,
        validation_split=0.2
    )

    return history