"""
Configuration settings for the Customer Churn Prediction project.
"""
import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directory
MODELS_DIR = ROOT_DIR / "models"

# Reports directory
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Source code directory
SRC_DIR = ROOT_DIR / "src"

# API directory
API_DIR = ROOT_DIR / "api"

# App directory
APP_DIR = ROOT_DIR / "app"

# Data file paths
RAW_DATA_FILE = RAW_DATA_DIR / "telco_customer_churn.csv"
PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
PROCESSED_TEST_FILE = PROCESSED_DATA_DIR / "test.csv"

# Model file paths
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
MODEL_METRICS_PATH = MODELS_DIR / "model_metrics.json"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.json"

# Model training settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Target column
TARGET_COLUMN = "Churn"

# Feature columns configuration
NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# All features
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Streamlit settings
STREAMLIT_HOST = os.getenv("STREAMLIT_HOST", "0.0.0.0")
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 8501))

# FastAPI endpoint for Streamlit
FASTAPI_URL = os.getenv("FASTAPI_URL", f"http://localhost:{API_PORT}")

# Model hyperparameters
LOGISTIC_REGRESSION_PARAMS = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l2"],
    "solver": ["lbfgs"],
    "max_iter": [1000],
    "random_state": [RANDOM_STATE],
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "random_state": [RANDOM_STATE],
}

XGBOOST_PARAMS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "random_state": [RANDOM_STATE],
}

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = ROOT_DIR / "logs" / "app.log"
