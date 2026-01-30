import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.preprocessing import DataPreprocessor, split_data

@pytest.fixture(scope="module")
def sample_data():
    rng = np.random.default_rng(42)
    n_samples = 100
    
    data = {
        "customerID": [f"CUST_{i:05d}" for i in range(n_samples)],
        "gender": rng.choice(["Male", "Female"], n_samples),
        "SeniorCitizen": rng.choice([0, 1], n_samples),
        "Partner": rng.choice(["Yes", "No"], n_samples),
        "Dependents": rng.choice(["Yes", "No"], n_samples),
        "tenure": rng.integers(0, 73, n_samples),
        "PhoneService": rng.choice(["Yes", "No"], n_samples),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n_samples),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n_samples),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n_samples),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n_samples),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n_samples),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n_samples),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n_samples),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n_samples),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n_samples),
        "PaperlessBilling": rng.choice(["Yes", "No"], n_samples),
        "PaymentMethod": rng.choice([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], n_samples),
        "MonthlyCharges": np.round(rng.uniform(18, 119, n_samples), 2),
        "TotalCharges": np.round(rng.uniform(18, 8500, n_samples), 2),
        "Churn": rng.choice(["Yes", "No"], n_samples),
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def preprocessor():
    return DataPreprocessor()

def test_clean_data_removes_customer_id(preprocessor, sample_data):
    cleaned = preprocessor.clean_data(sample_data.copy())
    assert "customerID" not in cleaned.columns
    assert "customerID" in sample_data.columns

def test_clean_data_converts_senior_citizen(preprocessor, sample_data):
    cleaned = preprocessor.clean_data(sample_data.copy())
    unique_vals = set(cleaned["SeniorCitizen"].unique())
    assert unique_vals.issubset({"Yes", "No"})

@pytest.mark.parametrize("expected_col", ["TenureGroup", "NumServices"])
def test_engineer_features_adds_columns(preprocessor, sample_data, expected_col):
    cleaned = preprocessor.clean_data(sample_data.copy())
    engineered = preprocessor.engineer_features(cleaned)
    assert expected_col in engineered.columns

def test_prepare_data_returns_arrays(preprocessor, sample_data):
    cleaned = preprocessor.clean_data(sample_data.copy())
    engineered = preprocessor.engineer_features(cleaned)
    X, y = preprocessor.prepare_data(engineered, fit=True)
    
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.dtype.kind in 'fiu'
    assert len(X) == len(sample_data)

def test_prepare_data_correct_shape(preprocessor, sample_data):
    cleaned = preprocessor.clean_data(sample_data.copy())
    engineered = preprocessor.engineer_features(cleaned)
    X, y = preprocessor.prepare_data(engineered, fit=True)
    
    assert X.shape == (len(sample_data), X.shape[1])
    assert y.shape == (len(sample_data),)

def test_split_data_maintains_proportions(preprocessor, sample_data):
    cleaned = preprocessor.clean_data(sample_data.copy())
    engineered = preprocessor.engineer_features(cleaned)
    X, y = preprocessor.prepare_data(engineered, fit=True)
    
    test_size = 0.2
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
    
    expected_test_len = int(len(X) * test_size)
    assert len(X_test) == expected_test_len
    assert len(X_train) == len(X) - expected_test_len

def test_preprocessor_get_feature_names(preprocessor, sample_data):
    cleaned = preprocessor.clean_data(sample_data.copy())
    engineered = preprocessor.engineer_features(cleaned)
    X, _ = preprocessor.prepare_data(engineered, fit=True)
    
    feature_names = preprocessor.get_feature_names()
    assert isinstance(feature_names, (list, np.ndarray))
    assert len(feature_names) == X.shape[1]