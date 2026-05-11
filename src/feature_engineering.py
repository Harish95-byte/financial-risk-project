def create_features(df):

    # Difference in balance
    df["balance_diff"] = (
        df["oldbalanceOrg"] - df["newbalanceOrig"]
    )

    # Large transaction detection
    df["large_transaction"] = (
        df["amount"] > 200000
    ).astype(int)

    # Detect account drained completely
    df["account_drained"] = (
        df["newbalanceOrig"] == 0
    ).astype(int)

    # Detect late-stage transaction behavior
    df["high_step"] = (
        df["step"] > 500
    ).astype(int)
    # Detect rapid transaction behavior
    df["rapid_transaction"] = (
        df["amount"] > 50000
    ).astype(int)
    # Detect risky transaction patterns
    df["high_risk_type"] = (
        (df["type"] == 1) | (df["type"] == 4)
    ).astype(int)

    return df