from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.train_model import train_model
from src.sequence_builder import create_sequences
from sklearn.preprocessing import MinMaxScaler
from src.train_lstm import (
    build_lstm_model,
    train_lstm_model
)
from src.risk_scoring import calculate_risk_score

def scale_features(feature_data):

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(
        feature_data
    )

    return scaled_data

DATA_PATH = "data/PS_20174392719_1491204439457_log.csv"


def main():

    # Load data
    df = load_data(DATA_PATH)

    # Preprocessing
    df = preprocess_data(df)

    # Feature engineering
    df = create_features(df)

    print("\nAfter Feature Engineering:")
    print(df.head())

    # Train model
    model = train_model(df)
    # Create sequence data
    # Features for sequence learning
    sequence_features = [
        "amount",
        "type",
        "balance_diff",
        "large_transaction",
        "account_drained",
        "rapid_transaction",
        "high_risk_type"
    ]

    # Prepare feature matrix
    feature_data = df[sequence_features].values
    # Scale sequence features
    feature_data = scale_features(feature_data)

    # Labels
    label_data = df["isFraud"].values

    # Create behavioral sequences
    X_seq, y_seq = create_sequences(
        feature_data,
        label_data
    )

    print("\nSequence Shape:", X_seq.shape)
    print("Labels Shape:", y_seq.shape)
    # Build LSTM model
    lstm_model = build_lstm_model(
        (X_seq.shape[1], X_seq.shape[2])
    )

    print("\nLSTM Model Created Successfully")
    # Train LSTM model
    history = train_lstm_model(
        lstm_model,
        X_seq,
        y_seq
    )

    print("\nLSTM Training Completed")
    # Save trained LSTM model
    lstm_model.save(
        "models/lstm_fraud_model.h5"
    )

    print("\nLSTM Model Saved Successfully")
    # Generate risk predictions
    predictions = lstm_model.predict(X_seq[:10])

    # Calculate behavioral risk scores
    risk_scores = calculate_risk_score(
        predictions
    )

    print("\nBehavioral Risk Scores:")
    print(risk_scores)

if __name__ == "__main__":
    main()