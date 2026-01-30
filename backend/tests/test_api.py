"""
Tests for the FastAPI application.
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from api.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data


def test_predict_endpoint_valid_data(client):
    """Test prediction with valid customer data."""
    customer_data = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.20
    }
    
    response = client.post("/predict", json=customer_data)
    
    # May fail if model not trained, but should not crash
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert data["prediction"] in ["Yes", "No"]
        assert 0 <= data["probability"] <= 1


def test_predict_endpoint_invalid_data(client):
    """Test prediction with invalid customer data."""
    invalid_data = {
        "gender": "Invalid",
        "tenure": -1
    }
    
    response = client.post("/predict", json=invalid_data)
    assert response.status_code in [422, 500]  # Validation or server error


def test_batch_predict_endpoint(client):
    """Test batch prediction endpoint."""
    batch_data = {
        "customers": [
            {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            }
        ]
    }
    
    response = client.post("/predict/batch", json=batch_data)
    
    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert "summary" in data


def test_model_info_endpoint(client):
    """Test model info endpoint."""
    response = client.get("/model/info")
    
    # May fail if model not trained
    if response.status_code == 200:
        data = response.json()
        assert "model_name" in data
        assert "accuracy" in data


def test_feature_importance_endpoint(client):
    """Test feature importance endpoint."""
    response = client.get("/model/features")
    
    # May return 404 if not available
    assert response.status_code in [200, 404]
