from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import sys

sys.path.append("churn_project/src")

from model_utils import predict_customer_churn


app = FastAPI(
    title="Enterprise Customer Churn Prediction API",
    description="API for predicting telecommunications customer churn risk.",
    version="1.0.0"
)


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    AvgMonthlySpend: float
    TotalServicesActive: int


@app.get("/")
def home():
    return {
        "status": "success",
        "message": "Enterprise Customer Churn Prediction API is running",
        "version": "1.0.0"
    }


@app.post("/predict")
def predict(customer: CustomerData):

    try:
        customer_dict = customer.model_dump()

        customer_df = pd.DataFrame([customer_dict])

        result = predict_customer_churn(customer_df)

        return {
            "status": "success",
            "prediction": result
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
