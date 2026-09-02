from pathlib import Path

# ==========================================================
# PROJECT ROOT
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

# ==========================================================
# DATA PATHS
# ==========================================================

DATA_DIR = ROOT_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_FILE = RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Processed Training Dataset
TRAINING_DATA_FILE = PROCESSED_DATA_DIR / "training_data.csv"

# ==========================================================
# MODEL PATHS
# ==========================================================

MODEL_DIR = ROOT_DIR / "models"

MODEL_FILE = MODEL_DIR / "customer_churn_model.pkl"

PREPROCESSOR_FILE = MODEL_DIR / "preprocessor.pkl"

# ==========================================================
# LOGS
# ==========================================================

LOG_DIR = ROOT_DIR / "logs"

LOG_FILE = LOG_DIR / "application.log"

# ==========================================================
# RANDOM SEED
# ==========================================================

RANDOM_STATE = 42

TEST_SIZE = 0.2

MODEL_VERSION = "1.0.0"