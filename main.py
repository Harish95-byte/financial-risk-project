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
from src.explainability import explain_model
from src.visualization import (
    plot_fraud_distribution,
    plot_shap_summary,
    plot_risk_evolution
)
from src.intent_drift import detect_intent_drift
from src.realtime_predictor import predict_realtime_risk

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
    model, X_train = train_model(df)
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
    # Generate SHAP explanations
    shap_values = explain_model(
        model,
        X_train[:100]
    )

    print("\nSHAP Explainability Generated")
    # Visualize fraud distribution
    plot_fraud_distribution(df)
    # SHAP feature importance visualization
    plot_shap_summary(
        shap_values,
        X_train[:100]
    )
    # Detect behavioral intent drift
    drift_scores = detect_intent_drift(
        risk_scores
    )

    print("\nIntent Drift Scores:")
    print(drift_scores)
    # Visualize behavioral risk evolution
    plot_risk_evolution(
        risk_scores
    )
    # Real-time fraud prediction
    sample_sequence = X_seq[0]

    realtime_risk = predict_realtime_risk(
        lstm_model,
        sample_sequence
    )

    print("\nReal-Time Fraud Risk:")
    print(realtime_risk)

if __name__ == "__main__":
    main()