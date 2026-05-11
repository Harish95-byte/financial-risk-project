import numpy as np


def calculate_risk_score(predictions):

    risk_scores = []

    for prediction in predictions:

        score = float(prediction[0])

        risk_scores.append(score)

    return np.array(risk_scores)