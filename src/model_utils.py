import joblib
import pandas as pd


MODEL_PATH = "churn_project/models/churn_prediction_pipeline.joblib"


def load_model():
    """
    Load the trained customer churn prediction pipeline.
    """
    return joblib.load(MODEL_PATH)


def predict_customer_churn(customer_data):
    """
    Predict churn probability and risk level for one customer.

    Parameters
    ----------
    customer_data : pandas.DataFrame
        Customer information containing the features expected by the model.

    Returns
    -------
    dict
        Churn probability, prediction, risk level, and recommended action.
    """

    model = load_model()

    probability = float(model.predict_proba(customer_data)[:, 1][0])
    prediction = int(model.predict(customer_data)[0])

    churn_prediction = "Yes" if prediction == 1 else "No"

    if probability >= 0.70:
        risk_level = "High Risk"
        recommended_action = "Immediate Retention Action"
    elif probability >= 0.50:
        risk_level = "Medium Risk"
        recommended_action = "Monitor and Engage"
    else:
        risk_level = "Low Risk"
        recommended_action = "Normal Customer Management"

    return {
        "churn_probability": round(probability, 4),
        "churn_prediction": churn_prediction,
        "risk_level": risk_level,
        "recommended_action": recommended_action
    }
