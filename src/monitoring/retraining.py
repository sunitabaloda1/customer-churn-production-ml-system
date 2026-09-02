"""
Retraining Trigger Logic

This module contains simple production-style rules that determine
whether the machine learning model should be retrained.
"""


AUC_THRESHOLD = 0.80
NEW_DATA_THRESHOLD = 5000
MISSING_VALUE_THRESHOLD = 0.05


def should_retrain(new_rows, current_auc, missing_ratio, drift_detected):
    """
    Returns True if retraining should be triggered.
    """

    if drift_detected:
        return True

    if current_auc < AUC_THRESHOLD:
        return True

    if new_rows >= NEW_DATA_THRESHOLD:
        return True

    if missing_ratio > MISSING_VALUE_THRESHOLD:
        return True

    return False


if __name__ == "__main__":

    retrain = should_retrain(
        new_rows=6200,
        current_auc=0.78,
        missing_ratio=0.02,
        drift_detected=False
    )

    if retrain:
        print("Retraining Triggered")
    else:
        print("No Retraining Required")