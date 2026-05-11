import matplotlib.pyplot as plt
import seaborn as sns
import shap

def plot_fraud_distribution(df):

    sns.countplot(
        x="isFraud",
        data=df
    )

    plt.title("Fraud Distribution")

    plt.show()
def plot_shap_summary(shap_values, X_train):

    shap.summary_plot(
        shap_values,
        X_train,
        show=True
    )
def plot_risk_evolution(risk_scores):

    plt.figure(figsize=(10, 5))

    plt.plot(
        risk_scores,
        marker="o"
    )

    plt.title(
        "Behavioral Risk Evolution"
    )

    plt.xlabel("Transaction Sequence")

    plt.ylabel("Risk Score")

    plt.grid(True)

    plt.show()