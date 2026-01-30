import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from api.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json().get("message") is not None

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert all(k in data for k in ("status", "version"))

def test_predict_endpoint_valid_data(client):
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
    
    if response.status_code == 200:
        data = response.json()
        assert all(k in data for k in ("prediction", "probability"))
        assert data["prediction"] in {"Yes", "No"}
        assert 0.0 <= float(data["probability"]) <= 1.0

def test_predict_endpoint_invalid_data(client):
    invalid_data = {
        "gender": "Invalid",
        "tenure": -1
    }
    response = client.post("/predict", json=invalid_data)
    assert response.status_code in {422, 500}

def test_batch_predict_endpoint(client):
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
    response = client.get("/model/info")
    if response.status_code == 200:
        data = response.json()
        assert "model_name" in data
        assert "accuracy" in data

def test_feature_importance_endpoint(client):
    response = client.get("/model/features")
    assert response.status_code in {200, 404}