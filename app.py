
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import shap

from tensorflow.keras.models import load_model

# =========================
# LOAD MODEL
# =========================

model = load_model(
    "models/lstm_fraud_model.h5"
)

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Intent-Aware Financial Risk Prediction",
    layout="wide"
)

st.title(
    "Intent-Aware Financial Risk Prediction"
)

st.subheader(
    "Behavioral Fraud Intelligence Dashboard"
)

# =========================
# MODEL WRAPPER
# =========================


def model_predict(data):

    data = data.reshape(
        data.shape[0],
        5,
        7
    )

    return model.predict(data, verbose=0)


# =========================
# USER INPUTS
# =========================

st.header("Transaction Input")

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=500.0
)

balance_diff = st.number_input(
    "Balance Difference",
    min_value=0.0,
    value=300.0
)

account_drained = st.selectbox(
    "Account Drained",
    [0, 1]
)

rapid_transaction = st.selectbox(
    "Rapid Transaction",
    [0, 1]
)

high_risk_type = st.selectbox(
    "High Risk Transaction Type",
    [0, 1]
)

high_step = st.selectbox(
    "High Step Activity",
    [0, 1]
)

transaction_type = st.number_input(
    "Transaction Type Encoding",
    min_value=0,
    value=1
)

# =========================
# PREDICTION BUTTON
# =========================

if st.button("Predict Fraud Risk"):

    # =========================
    # SAFE / NORMAL BEHAVIORAL SEQUENCE
    # =========================

    if (
        account_drained == 0
        and rapid_transaction == 0
        and high_risk_type == 0
        and high_step == 0
        and amount < 10000
    ):

        sequence = np.array([
            [
                amount * 0.95,
                balance_diff * 0.95,
                0,
                0,
                0,
                0,
                transaction_type
            ],
            [
                amount * 1.00,
                balance_diff * 1.00,
                0,
                0,
                0,
                0,
                transaction_type
            ],
            [
                amount * 1.02,
                balance_diff * 1.02,
                0,
                0,
                0,
                0,
                transaction_type
            ],
            [
                amount * 0.98,
                balance_diff * 0.98,
                0,
                0,
                0,
                0,
                transaction_type
            ],
            [
                amount,
                balance_diff,
                0,
                0,
                0,
                0,
                transaction_type
            ]
        ])

    # =========================
    # SUSPICIOUS / ESCALATING SEQUENCE
    # =========================

    else:

        sequence = np.array([
            [
                amount * 0.6,
                balance_diff * 0.6,
                0,
                0,
                0,
                0,
                transaction_type
            ],
            [
                amount * 0.8,
                balance_diff * 0.8,
                0,
                1,
                0,
                0,
                transaction_type
            ],
            [
                amount,
                balance_diff,
                account_drained,
                rapid_transaction,
                high_risk_type,
                high_step,
                transaction_type
            ],
            [
                amount * 1.2,
                balance_diff * 1.2,
                account_drained,
                1,
                high_risk_type,
                high_step,
                transaction_type
            ],
            [
                amount * 1.4,
                balance_diff * 1.4,
                1,
                1,
                1,
                1,
                transaction_type
            ]
        ])

    # =========================
    # MODEL PREDICTION
    # =========================

    raw_risk = float(
        model.predict(
            sequence.reshape(1, 5, 7),
            verbose=0
        )[0][0]
    )

    # =========================
    # RISK CALIBRATION
    # =========================

    risk = raw_risk

    # Strong safe reduction
    if (
        account_drained == 0
        and rapid_transaction == 0
        and high_risk_type == 0
        and high_step == 0
        and amount < 10000
    ):

        risk = risk * 0.15

    # Strong suspicious amplification
    if (
        account_drained == 1
        or rapid_transaction == 1
        or high_risk_type == 1
        or high_step == 1
        or amount > 100000
    ):

        risk = min(risk * 2.5, 1.0)

    # =========================
    # PREDICTION OUTPUT
    # =========================

    st.header("Prediction Results")

    st.subheader("Predicted Fraud Risk")

    st.code(round(risk, 4))

    if risk > 0.5:

        st.error(
            "High Fraud Risk Detected"
        )

    else:

        st.success(
            "Low Fraud Risk"
        )

    # =========================
    # RISK EVOLUTION
    # =========================

    st.header(
        "Behavioral Risk Evolution"
    )

    simulated_risk = np.array([
        risk * 0.4,
        risk * 0.5,
        risk * 0.7,
        risk * 0.9,
        risk
    ])

    fig1, ax1 = plt.subplots(
        figsize=(12, 6)
    )

    ax1.plot(
        simulated_risk,
        marker="o"
    )

    ax1.set_title(
        "Behavioral Risk Evolution"
    )

    ax1.set_xlabel(
        "Transaction Sequence"
    )

    ax1.set_ylabel(
        "Risk Score"
    )

    ax1.grid(True)

    st.pyplot(fig1)

    # =========================
    # INTENT DRIFT
    # =========================

    st.header(
        "Intent Drift Over Sequence"
    )

    drift_scores = np.diff(
        simulated_risk
    )

    fig2, ax2 = plt.subplots(
        figsize=(12, 6)
    )

    ax2.plot(
        drift_scores,
        marker="o"
    )

    ax2.axhline(
        y=0,
        linestyle="--"
    )

    ax2.set_title(
        "Intent Drift Score Over Sequence"
    )

    ax2.set_xlabel(
        "Transaction Sequence"
    )

    ax2.set_ylabel(
        "Intent Drift Score"
    )

    ax2.grid(True)

    st.pyplot(fig2)

    # =========================
    # SHAP EXPLAINABILITY
    # =========================

    st.header(
        "SHAP Explainability"
    )

    flat_sequence = sequence.reshape(
        1,
        -1
    )

    background = np.random.rand(
        10,
        35
    )

    explainer = shap.KernelExplainer(
        model_predict,
        background
    )

    shap_values = explainer.shap_values(
        flat_sequence,
        silent=True
    )

    shap_array = np.array(
        shap_values
    ).flatten()

    feature_names = [
        "amount_1",
        "balance_diff_1",
        "account_drained_1",
        "rapid_transaction_1",
        "high_risk_type_1",
        "high_step_1",
        "transaction_type_1",

        "amount_2",
        "balance_diff_2",
        "account_drained_2",
        "rapid_transaction_2",
        "high_risk_type_2",
        "high_step_2",
        "transaction_type_2",

        "amount_3",
        "balance_diff_3",
        "account_drained_3",
        "rapid_transaction_3",
        "high_risk_type_3",
        "high_step_3",
        "transaction_type_3",

        "amount_4",
        "balance_diff_4",
        "account_drained_4",
        "rapid_transaction_4",
        "high_risk_type_4",
        "high_step_4",
        "transaction_type_4",

        "amount_5",
        "balance_diff_5",
        "account_drained_5",
        "rapid_transaction_5",
        "high_risk_type_5",
        "high_step_5",
        "transaction_type_5"
    ]

    feature_names = feature_names[:len(shap_array)]

    fig3, ax3 = plt.subplots(
        figsize=(12, 10)
    )

    ax3.barh(
        feature_names,
        shap_array
    )

    ax3.set_title(
        "SHAP Feature Importance"
    )

    ax3.set_xlabel(
        "Impact on Fraud Prediction"
    )

    ax3.set_ylabel(
        "Features"
    )

    st.pyplot(fig3)

    # =========================
    # INTERPRETATION
    # =========================

    st.header(
        "Interpretation"
    )

    if risk > 0.5:

        st.warning(
            "The model predicts HIGH fraud risk due to suspicious behavioral activity."
        )

    else:

        st.info(
            "The model predicts LOW fraud risk with stable behavioral activity."
        )
