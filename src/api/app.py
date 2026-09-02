from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

from src.config.config import (
    MODEL_FILE,
    MODEL_VERSION
)

# --------------------------------------------------------
# Load trained model
# --------------------------------------------------------

model = joblib.load(MODEL_FILE)

# --------------------------------------------------------
# Create FastAPI app
# --------------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    version=MODEL_VERSION
)

# --------------------------------------------------------
# Input Schema
# --------------------------------------------------------

class Customer(BaseModel):

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

# --------------------------------------------------------
# Root Endpoint
# --------------------------------------------------------

@app.get("/")

def home():

    return {
        "message": "Customer Churn Prediction API",
        "model_version": MODEL_VERSION
    }

# --------------------------------------------------------
# Health Check
# --------------------------------------------------------

@app.get("/health")

def health():

    return {
        "status": "healthy"
    }

# --------------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------------

@app.post("/predict")

def predict(customer: Customer):

    df = pd.DataFrame([customer.dict()])

    # Feature Engineering (same as training)

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    df["TotalServices"] = (
        df[service_columns]
        .replace("No internet service", "No")
        .replace("No phone service", "No")
        .eq("Yes")
        .sum(axis=1)
    )

    df["AvgMonthlySpend"] = (
        df["TotalCharges"] /
        df["tenure"].replace(0, 1)
    )

    df["IsLongTermCustomer"] = (
        df["tenure"] >= 24
    ).astype(int)

    df["HasAutoPayment"] = (
        df["PaymentMethod"]
        .str.contains("automatic", case=False)
        .astype(int)
    )

    df["FiberCustomer"] = (
        df["InternetService"] == "Fiber optic"
    ).astype(int)

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 4),
        "model_version": MODEL_VERSION
    }