import os
import pandas as pd

from src.config.config import (
    RAW_DATA_FILE,
    TRAINING_DATA_FILE
)
from src.monitoring.logger import logger
from src.preprocessing.preprocess import preprocess_data


EXPECTED_MONTHLY_CHARGES_MEAN = 64.76
DRIFT_THRESHOLD = 0.10


def load_data():

    logger.info("Loading dataset...")

    df = pd.read_csv(RAW_DATA_FILE)

    logger.info(
        f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns."
    )

    return df


def ingest_training_data(df):
    """
    Simulates a batch ingestion process by creating or appending
    data to a processed training dataset.
    """

    os.makedirs(TRAINING_DATA_FILE.parent, exist_ok=True)

    if TRAINING_DATA_FILE.exists():

        existing_df = pd.read_csv(TRAINING_DATA_FILE)

        combined_df = pd.concat([existing_df, df], ignore_index=True)

        combined_df.to_csv(TRAINING_DATA_FILE, index=False)

        logger.info(
            f"Appended {len(df)} new rows."
        )

        logger.info(
            f"Training dataset now contains {len(combined_df)} rows."
        )

    else:

        df.to_csv(TRAINING_DATA_FILE, index=False)

        logger.info(
            f"Created training dataset with {len(df)} rows."
        )


def run_data_quality_checks(df):

    logger.info("Running Data Quality Checks...")

    total_missing = df.isnull().sum().sum()

    if total_missing == 0:
        logger.info("PASS: No missing values detected.")
    else:
        logger.warning(
            f"WARNING: {total_missing} missing values detected."
        )

    current_mean = df["MonthlyCharges"].mean()

    drift = abs(
        current_mean - EXPECTED_MONTHLY_CHARGES_MEAN
    ) / EXPECTED_MONTHLY_CHARGES_MEAN

    logger.info(
        f"Expected MonthlyCharges Mean : {EXPECTED_MONTHLY_CHARGES_MEAN:.2f}"
    )

    logger.info(
        f"Current MonthlyCharges Mean  : {current_mean:.2f}"
    )

    if drift > DRIFT_THRESHOLD:

        logger.warning(
            f"WARNING: Data Drift Detected! Drift = {drift:.2%}"
        )

    else:

        logger.info(
            f"PASS: No significant drift detected. Drift = {drift:.2%}"
        )


if __name__ == "__main__":

    df = load_data()

    ingest_training_data(df)

    run_data_quality_checks(df)

    df = preprocess_data(df)

    print(df.head())

    print("\nDataset Shape:", df.shape)

    print("\nMissing Values:\n")

    print(df.isnull().sum())