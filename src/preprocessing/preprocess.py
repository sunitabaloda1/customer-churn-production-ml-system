import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from src.monitoring.logger import logger


def preprocess_data(df):
    """
    Clean dataset and create engineered features.
    """

    logger.info("Starting preprocessing...")

    df = df.copy()

    # ----------------------------------------
    # Data Cleaning
    # ----------------------------------------

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # ----------------------------------------
    # Feature Engineering
    # ----------------------------------------

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

    logger.info("Feature engineering completed.")

    return df


def build_preprocessor(df):

    categorical_columns = df.select_dtypes(
        include="object"
    ).columns.tolist()

    categorical_columns.remove("customerID")

    categorical_columns.remove("Churn")

    numeric_columns = [
        col
        for col in df.columns
        if col not in categorical_columns
        and col != "customerID"
        and col != "Churn"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            ),
            (
                "numeric",
                "passthrough",
                numeric_columns
            )
        ]
    )

    return preprocessor