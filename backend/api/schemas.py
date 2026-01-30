"""
Pydantic schemas for API request and response validation.
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from enum import Enum


class GenderEnum(str, Enum):
    male = "Male"
    female = "Female"


class YesNoEnum(str, Enum):
    yes = "Yes"
    no = "No"


class InternetServiceEnum(str, Enum):
    dsl = "DSL"
    fiber_optic = "Fiber optic"
    no = "No"


class ContractEnum(str, Enum):
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class PaymentMethodEnum(str, Enum):
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"
    bank_transfer = "Bank transfer (automatic)"
    credit_card = "Credit card (automatic)"


class MultipleLinesEnum(str, Enum):
    yes = "Yes"
    no = "No"
    no_phone_service = "No phone service"


class InternetDependentEnum(str, Enum):
    yes = "Yes"
    no = "No"
    no_internet_service = "No internet service"


class CustomerData(BaseModel):
    """
    Schema for customer input data for churn prediction.
    """
    gender: GenderEnum = Field(..., description="Customer gender")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Whether customer is senior citizen (0 or 1)")
    Partner: YesNoEnum = Field(..., description="Whether customer has a partner")
    Dependents: YesNoEnum = Field(..., description="Whether customer has dependents")
    tenure: int = Field(..., ge=0, le=72, description="Number of months with the company")
    PhoneService: YesNoEnum = Field(..., description="Whether customer has phone service")
    MultipleLines: MultipleLinesEnum = Field(..., description="Whether customer has multiple lines")
    InternetService: InternetServiceEnum = Field(..., description="Type of internet service")
    OnlineSecurity: InternetDependentEnum = Field(..., description="Whether customer has online security")
    OnlineBackup: InternetDependentEnum = Field(..., description="Whether customer has online backup")
    DeviceProtection: InternetDependentEnum = Field(..., description="Whether customer has device protection")
    TechSupport: InternetDependentEnum = Field(..., description="Whether customer has tech support")
    StreamingTV: InternetDependentEnum = Field(..., description="Whether customer has streaming TV")
    StreamingMovies: InternetDependentEnum = Field(..., description="Whether customer has streaming movies")
    Contract: ContractEnum = Field(..., description="Contract type")
    PaperlessBilling: YesNoEnum = Field(..., description="Whether customer has paperless billing")
    PaymentMethod: PaymentMethodEnum = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges in dollars")
    TotalCharges: float = Field(..., ge=0, description="Total charges in dollars")
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }


class CustomerDataSimple(BaseModel):
    """
    Simplified schema for customer input (without enums for easier usage).
    """
    gender: str = Field(..., description="Customer gender (Male/Female)")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Whether customer is senior citizen (0 or 1)")
    Partner: str = Field(..., description="Whether customer has a partner (Yes/No)")
    Dependents: str = Field(..., description="Whether customer has dependents (Yes/No)")
    tenure: int = Field(..., ge=0, le=100, description="Number of months with the company")
    PhoneService: str = Field(..., description="Whether customer has phone service (Yes/No)")
    MultipleLines: str = Field(..., description="Multiple lines status")
    InternetService: str = Field(..., description="Type of internet service")
    OnlineSecurity: str = Field(..., description="Online security status")
    OnlineBackup: str = Field(..., description="Online backup status")
    DeviceProtection: str = Field(..., description="Device protection status")
    TechSupport: str = Field(..., description="Tech support status")
    StreamingTV: str = Field(..., description="Streaming TV status")
    StreamingMovies: str = Field(..., description="Streaming movies status")
    Contract: str = Field(..., description="Contract type")
    PaperlessBilling: str = Field(..., description="Paperless billing status")
    PaymentMethod: str = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges in dollars")
    TotalCharges: float = Field(..., ge=0, description="Total charges in dollars")
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """
    Schema for prediction response.
    """
    prediction: str = Field(..., description="Churn prediction (Yes/No)")
    probability: float = Field(..., description="Probability of churn (0-1)")
    confidence: str = Field(..., description="Confidence level (Low/Medium/High)")
    risk_factors: List[str] = Field(default=[], description="Identified risk factors")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "Yes",
                "probability": 0.75,
                "confidence": "High",
                "risk_factors": [
                    "Month-to-month contract",
                    "Short tenure (< 12 months)",
                    "Electronic check payment"
                ]
            }
        }


class BatchPredictionRequest(BaseModel):
    """
    Schema for batch prediction request.
    """
    customers: List[CustomerDataSimple] = Field(..., description="List of customer data for batch prediction")


class BatchPredictionResponse(BaseModel):
    """
    Schema for batch prediction response.
    """
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")


class ModelInfo(BaseModel):
    """
    Schema for model information.
    """
    model_name: str = Field(..., description="Name of the model")
    model_type: str = Field(..., description="Type of the model")
    accuracy: float = Field(..., description="Model accuracy")
    roc_auc: float = Field(..., description="ROC-AUC score")
    features: List[str] = Field(..., description="List of features used")
    last_trained: Optional[str] = Field(None, description="Last training date")


class FeatureImportance(BaseModel):
    """
    Schema for feature importance.
    """
    feature: str = Field(..., description="Feature name")
    importance: float = Field(..., description="Importance score")


class HealthCheck(BaseModel):
    """
    Schema for health check response.
    """
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")
