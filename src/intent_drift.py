import numpy as np


def detect_intent_drift(risk_scores):

    drift_scores = []

    for i in range(1, len(risk_scores)):

        drift = risk_scores[i] - risk_scores[i - 1]

        drift_scores.append(drift)

    return np.array(drift_scores)