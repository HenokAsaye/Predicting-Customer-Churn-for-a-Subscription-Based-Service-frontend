"""
Tests for the preprocessing module.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import DataPreprocessor, split_data


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        "customerID": [f"CUST_{i:05d}" for i in range(n_samples)],
        "gender": np.random.choice(["Male", "Female"], n_samples),
        "SeniorCitizen": np.random.choice([0, 1], n_samples),
        "Partner": np.random.choice(["Yes", "No"], n_samples),
        "Dependents": np.random.choice(["Yes", "No"], n_samples),
        "tenure": np.random.randint(0, 73, n_samples),
        "PhoneService": np.random.choice(["Yes", "No"], n_samples),
        "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n_samples),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n_samples),
        "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n_samples),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n_samples),
        "PaymentMethod": np.random.choice([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], n_samples),
        "MonthlyCharges": np.round(np.random.uniform(18, 119, n_samples), 2),
        "TotalCharges": np.round(np.random.uniform(18, 8500, n_samples), 2),
        "Churn": np.random.choice(["Yes", "No"], n_samples),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def preprocessor():
    """Create a DataPreprocessor instance."""
    return DataPreprocessor()


def test_clean_data_removes_customer_id(preprocessor, sample_data):
    """Test that clean_data removes customerID column."""
    cleaned = preprocessor.clean_data(sample_data)
    assert "customerID" not in cleaned.columns


def test_clean_data_converts_senior_citizen(preprocessor, sample_data):
    """Test that clean_data converts SeniorCitizen to Yes/No."""
    cleaned = preprocessor.clean_data(sample_data)
    assert set(cleaned["SeniorCitizen"].unique()).issubset({"Yes", "No"})


def test_engineer_features_adds_tenure_group(preprocessor, sample_data):
    """Test that engineer_features adds TenureGroup."""
    cleaned = preprocessor.clean_data(sample_data)
    engineered = preprocessor.engineer_features(cleaned)
    assert "TenureGroup" in engineered.columns


def test_engineer_features_adds_num_services(preprocessor, sample_data):
    """Test that engineer_features adds NumServices."""
    cleaned = preprocessor.clean_data(sample_data)
    engineered = preprocessor.engineer_features(cleaned)
    assert "NumServices" in engineered.columns


def test_prepare_data_returns_arrays(preprocessor, sample_data):
    """Test that prepare_data returns numpy arrays."""
    cleaned = preprocessor.clean_data(sample_data)
    engineered = preprocessor.engineer_features(cleaned)
    X, y = preprocessor.prepare_data(engineered, fit=True)
    
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_prepare_data_correct_shape(preprocessor, sample_data):
    """Test that prepare_data returns correct shapes."""
    cleaned = preprocessor.clean_data(sample_data)
    engineered = preprocessor.engineer_features(cleaned)
    X, y = preprocessor.prepare_data(engineered, fit=True)
    
    assert X.shape[0] == len(sample_data)
    assert y.shape[0] == len(sample_data)


def test_split_data_maintains_proportions(preprocessor, sample_data):
    """Test that split_data maintains correct proportions."""
    cleaned = preprocessor.clean_data(sample_data)
    engineered = preprocessor.engineer_features(cleaned)
    X, y = preprocessor.prepare_data(engineered, fit=True)
    
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    
    assert len(X_train) == pytest.approx(len(X) * 0.8, abs=2)
    assert len(X_test) == pytest.approx(len(X) * 0.2, abs=2)


def test_preprocessor_get_feature_names(preprocessor, sample_data):
    """Test that get_feature_names returns feature names."""
    cleaned = preprocessor.clean_data(sample_data)
    engineered = preprocessor.engineer_features(cleaned)
    X, y = preprocessor.prepare_data(engineered, fit=True)
    
    feature_names = preprocessor.get_feature_names()
    assert len(feature_names) > 0
    assert X.shape[1] == len(feature_names)
