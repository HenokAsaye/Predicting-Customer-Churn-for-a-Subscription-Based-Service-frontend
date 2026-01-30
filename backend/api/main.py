"""
FastAPI application for Customer Churn Prediction API.
Serves the trained model for making predictions.
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent))

from api.schemas import (
    CustomerDataSimple,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    FeatureImportance,
    HealthCheck,
)
from config import (
    BEST_MODEL_PATH,
    PREPROCESSOR_PATH,
    MODEL_METRICS_PATH,
    FEATURE_IMPORTANCE_PATH,
    API_HOST,
    API_PORT,
)
from src.preprocessing import DataPreprocessor


# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="""
    ## Customer Churn Prediction API
    
    This API provides endpoints for predicting customer churn using machine learning.
    
    ### Features:
    - **Single Prediction**: Predict churn for a single customer
    - **Batch Prediction**: Predict churn for multiple customers at once
    - **Model Information**: Get details about the trained model
    - **Feature Importance**: View which features contribute most to predictions
    
    ### Usage:
    1. Use the `/predict` endpoint for single customer predictions
    2. Use the `/predict/batch` endpoint for multiple customers
    3. Check `/model/info` for model performance metrics
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessor
model = None
preprocessor = None
model_metrics = None
feature_importance = None


def load_model_and_preprocessor():
    """Load the trained model and preprocessor."""
    global model, preprocessor, model_metrics, feature_importance
    
    try:
        # Load model
        if BEST_MODEL_PATH.exists():
            model = joblib.load(BEST_MODEL_PATH)
            logger.info(f"Model loaded from {BEST_MODEL_PATH}")
        else:
            logger.warning(f"Model not found at {BEST_MODEL_PATH}")
            model = None
        
        # Load preprocessor
        if PREPROCESSOR_PATH.exists():
            preprocessor = DataPreprocessor()
            preprocessor.load_preprocessor()
            logger.info(f"Preprocessor loaded from {PREPROCESSOR_PATH}")
        else:
            logger.warning(f"Preprocessor not found at {PREPROCESSOR_PATH}")
            preprocessor = None
        
        # Load metrics
        if MODEL_METRICS_PATH.exists():
            with open(MODEL_METRICS_PATH, "r") as f:
                model_metrics = json.load(f)
            logger.info("Model metrics loaded")
        
        # Load feature importance
        if FEATURE_IMPORTANCE_PATH.exists():
            with open(FEATURE_IMPORTANCE_PATH, "r") as f:
                feature_importance = json.load(f)
            logger.info("Feature importance loaded")
            
    except Exception as e:
        logger.error(f"Error loading model/preprocessor: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting Customer Churn Prediction API...")
    load_model_and_preprocessor()
    logger.info("API startup complete")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Customer Churn Prediction API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Check API health status."""
    return HealthCheck(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0"
    )


def identify_risk_factors(customer_data: Dict[str, Any]) -> List[str]:
    """Identify risk factors for churn based on customer data."""
    risk_factors = []
    
    # Contract type
    if customer_data.get("Contract") == "Month-to-month":
        risk_factors.append("Month-to-month contract (highest churn risk)")
    
    # Tenure
    tenure = customer_data.get("tenure", 0)
    if tenure < 12:
        risk_factors.append(f"Short tenure ({tenure} months < 12)")
    
    # Payment method
    if customer_data.get("PaymentMethod") == "Electronic check":
        risk_factors.append("Electronic check payment method")
    
    # Internet service
    if customer_data.get("InternetService") == "Fiber optic":
        risk_factors.append("Fiber optic service (higher monthly charges)")
    
    # Tech support
    if customer_data.get("TechSupport") == "No":
        risk_factors.append("No tech support subscription")
    
    # Online security
    if customer_data.get("OnlineSecurity") == "No":
        risk_factors.append("No online security subscription")
    
    # Monthly charges
    monthly = customer_data.get("MonthlyCharges", 0)
    if monthly > 70:
        risk_factors.append(f"High monthly charges (${monthly:.2f})")
    
    # Paperless billing
    if customer_data.get("PaperlessBilling") == "Yes":
        risk_factors.append("Paperless billing enabled")
    
    return risk_factors


def get_confidence_level(probability: float) -> str:
    """Get confidence level based on probability."""
    if probability < 0.3 or probability > 0.7:
        return "High"
    elif probability < 0.4 or probability > 0.6:
        return "Medium"
    else:
        return "Low"


def preprocess_customer_data(customer_data: Dict[str, Any]) -> np.ndarray:
    """Preprocess customer data for prediction."""
    try:
        # Convert SeniorCitizen to Yes/No for preprocessing
        data = customer_data.copy()
        data["SeniorCitizen"] = "Yes" if data["SeniorCitizen"] == 1 else "No"
        
        # Ensure numeric fields are actually numeric
        for numeric_col in ["tenure", "MonthlyCharges", "TotalCharges"]:
            if numeric_col in data:
                data[numeric_col] = float(data[numeric_col])
        
        # Create DataFrame
        df = pd.DataFrame([data])
        logger.debug(f"Created DataFrame with columns: {df.columns.tolist()}")
        logger.debug(f"DataFrame dtypes: {df.dtypes.to_dict()}")
        
        # Clean and engineer features
        df_clean = preprocessor.clean_data(df)
        logger.debug(f"After cleaning: {df_clean.columns.tolist()}")
        
        df_engineered = preprocessor.engineer_features(df_clean)
        logger.debug(f"After engineering: {df_engineered.columns.tolist()}")
        
        # Transform - make sure we're getting the right data
        X, _ = preprocessor.prepare_data(df_engineered, fit=False)
        
        if X is None or len(X) == 0:
            raise ValueError("Preprocessed data is empty")
        
        logger.debug(f"Transformed data shape: {X.shape}")
        return X
        
    except Exception as e:
        logger.error(f"Error in preprocess_customer_data: {e}", exc_info=True)
        raise


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_churn(customer: CustomerDataSimple):
    """
    Predict customer churn for a single customer.
    
    - **customer**: Customer data including demographics, services, and account info
    
    Returns prediction result with probability and risk factors.
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Convert to dict
        customer_data = customer.model_dump()
        logger.debug(f"Processing prediction for customer data: {list(customer_data.keys())}")
        
        # Preprocess
        X = preprocess_customer_data(customer_data)
        
        # Validate preprocessed data
        if X is None or X.size == 0:
            raise ValueError("Preprocessing resulted in empty data")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        logger.debug(f"Preprocessed data shape for prediction: {X.shape}")
        
        # Predict
        prediction_proba = model.predict_proba(X)
        logger.debug(f"Prediction proba shape: {prediction_proba.shape}")
        
        if prediction_proba.shape[0] != 1:
            raise ValueError(f"Expected 1 prediction, got {prediction_proba.shape[0]}")
        
        probability = float(prediction_proba[0, 1])
        prediction = "Yes" if probability >= 0.5 else "No"
        
        # Get risk factors and confidence
        risk_factors = identify_risk_factors(customer_data) if prediction == "Yes" else []
        confidence = get_confidence_level(probability)
        
        logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}")
        
        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            confidence=confidence,
            risk_factors=risk_factors
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making prediction: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_churn_batch(request: BatchPredictionRequest):
    """
    Predict customer churn for multiple customers at once.
    
    - **customers**: List of customer data
    
    Returns predictions for all customers with summary statistics.
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        predictions = []
        churn_count = 0
        total_probability = 0
        
        for customer in request.customers:
            customer_data = customer.model_dump()
            
            # Preprocess
            X = preprocess_customer_data(customer_data)
            
            # Predict
            probability = model.predict_proba(X)[0, 1]
            prediction = "Yes" if probability >= 0.5 else "No"
            
            risk_factors = identify_risk_factors(customer_data) if prediction == "Yes" else []
            confidence = get_confidence_level(probability)
            
            predictions.append(PredictionResponse(
                prediction=prediction,
                probability=round(float(probability), 4),
                confidence=confidence,
                risk_factors=risk_factors
            ))
            
            if prediction == "Yes":
                churn_count += 1
            total_probability += probability
        
        # Summary statistics
        summary = {
            "total_customers": len(predictions),
            "predicted_churns": churn_count,
            "predicted_retained": len(predictions) - churn_count,
            "churn_rate": round(churn_count / len(predictions) * 100, 2) if predictions else 0,
            "average_churn_probability": round(total_probability / len(predictions), 4) if predictions else 0
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making predictions: {str(e)}"
        )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """
    Get information about the trained model.
    
    Returns model type, performance metrics, and features used.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded."
        )
    
    try:
        # Get model type
        model_type = type(model).__name__
        
        # Get metrics
        best_model_name = model_metrics.get("best_model", "Unknown") if model_metrics else "Unknown"
        
        if model_metrics and best_model_name in model_metrics:
            metrics = model_metrics[best_model_name]["test_metrics"]
            accuracy = metrics.get("accuracy", 0)
            roc_auc = metrics.get("roc_auc", 0)
        else:
            accuracy = 0
            roc_auc = 0
        
        # Get features
        features = preprocessor.get_feature_names() if preprocessor else []
        
        return ModelInfo(
            model_name=best_model_name,
            model_type=model_type,
            accuracy=round(accuracy, 4),
            roc_auc=round(roc_auc, 4),
            features=features,
            last_trained=datetime.now().strftime("%Y-%m-%d")
        )
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting model info: {str(e)}"
        )


@app.get("/model/features", response_model=List[FeatureImportance], tags=["Model"])
async def get_feature_importance():
    """
    Get feature importance scores from the trained model.
    
    Returns list of features with their importance scores, sorted by importance.
    """
    if feature_importance is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feature importance not available."
        )
    
    return [
        FeatureImportance(feature=f["feature"], importance=round(f["importance"], 6))
        for f in sorted(feature_importance, key=lambda x: x["importance"], reverse=True)
    ]


@app.get("/model/reload", tags=["Model"])
async def reload_model():
    """
    Reload the model and preprocessor from disk.
    
    Use this after retraining the model to load the updated version.
    """
    try:
        load_model_and_preprocessor()
        return {"message": "Model reloaded successfully", "model_loaded": model is not None}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reloading model: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )
