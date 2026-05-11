import shap


def explain_model(model, X_train):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_train)

    return shap_values