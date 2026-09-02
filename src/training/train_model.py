import os
import json
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from src.ingestion.ingest_data import load_data
from src.preprocessing.preprocess import (
    preprocess_data,
    build_preprocessor
)

from src.config.config import (
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_DIR,
    MODEL_FILE,
    PREPROCESSOR_FILE,
    MODEL_VERSION
)

from src.monitoring.logger import logger


def prepare_training_data():

    logger.info("Loading dataset...")

    df = load_data()

    df = preprocess_data(df)

    X = df.drop(columns=["customerID", "Churn"])

    y = df["Churn"].map(
        {
            "No": 0,
            "Yes": 1
        }
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    preprocessor = build_preprocessor(df)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )


def train_model(
    model,
    model_name,
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):

    logger.info(f"Training {model_name}")

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\n====================================")

    print(model_name)

    print("====================================")

    print(f"Accuracy : {accuracy:.4f}")

    print(f"ROC AUC  : {auc:.4f}")

    print(cm)

    return (
        pipeline,
        accuracy,
        auc,
        report
    )
def save_artifacts(best_pipeline, metrics):

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("artifacts/eval", exist_ok=True)

    # Save complete pipeline
    joblib.dump(best_pipeline, MODEL_FILE)

    # Save preprocessor separately
    joblib.dump(
        best_pipeline.named_steps["preprocessor"],
        PREPROCESSOR_FILE
    )

    with open(
        "artifacts/eval/metrics.json",
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    logger.info("Artifacts saved successfully.")


def main():

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_training_data()

    #########################################################
    # Baseline Model
    #########################################################

    baseline_pipeline, baseline_accuracy, baseline_auc, baseline_report = train_model(
        LogisticRegression(max_iter=1000),
        "Baseline - Logistic Regression",
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )

    #########################################################
    # Candidate Model
    #########################################################

    candidate_pipeline, candidate_accuracy, candidate_auc, candidate_report = train_model(
        RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE
        ),
        "Candidate - Random Forest",
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )

    #########################################################
    # Model Promotion Rule
    #########################################################

    PROMOTION_THRESHOLD = 0.80

    logger.info("Evaluating model promotion rule...")

    if (
        candidate_auc >= PROMOTION_THRESHOLD
        and candidate_auc >= baseline_auc
    ):

        logger.info("Candidate model promoted to production.")

        best_pipeline = candidate_pipeline
        best_model = "Random Forest"
        best_auc = candidate_auc

    else:

        logger.info("Baseline model retained.")

        best_pipeline = baseline_pipeline
        best_model = "Logistic Regression"
        best_auc = baseline_auc



    metrics = {
        "model_version": MODEL_VERSION,
        "best_model": best_model,
        "baseline_accuracy": baseline_accuracy,
        "baseline_auc": baseline_auc,
        "candidate_accuracy": candidate_accuracy,
        "candidate_auc": candidate_auc,
        "selected_model_auc": best_auc,
        "promotion_threshold": PROMOTION_THRESHOLD,
        "promotion_status": (
        "PROMOTED"
        if best_model == "Random Forest"
        else "BASELINE_RETAINED"
)
    }

    save_artifacts(
        best_pipeline,
        metrics
    )

    print("\n===================================")
    print("BEST MODEL")
    print("===================================")

    print(best_model)

    print(f"ROC AUC : {best_auc:.4f}")

    print("\nArtifacts saved successfully.")


if __name__ == "__main__":
    main()