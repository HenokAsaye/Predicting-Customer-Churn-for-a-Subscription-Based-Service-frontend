"""
Data loading utilities for the Customer Churn Prediction project.
Handles loading raw data and downloading from Kaggle if needed.
"""
import os
import pandas as pd
from pathlib import Path
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import RAW_DATA_FILE, RAW_DATA_DIR


def load_raw_data(filepath: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """
    Load raw data from CSV file.
    
    Args:
        filepath: Path to the raw data file.
        
    Returns:
        DataFrame containing the raw data.
    """
    if not filepath.exists():
        logger.warning(f"Data file not found at {filepath}")
        logger.info("Please download the Telco Customer Churn dataset from Kaggle:")
        logger.info("https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        logger.info(f"And save it to: {filepath}")
        
        # Create sample data for demonstration
        logger.info("Creating sample dataset for demonstration...")
        return create_sample_data()
    
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    return df


def create_sample_data(n_samples: int = 7043) -> pd.DataFrame:
    """
    Create a sample dataset similar to Telco Customer Churn dataset.
    This is used for demonstration when actual data is not available.
    
    Args:
        n_samples: Number of samples to generate.
        
    Returns:
        DataFrame with sample data.
    """
    import numpy as np
    
    np.random.seed(42)
    
    # Generate sample data
    data = {
        "customerID": [f"CUST_{i:05d}" for i in range(n_samples)],
        "gender": np.random.choice(["Male", "Female"], n_samples),
        "SeniorCitizen": np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
        "Partner": np.random.choice(["Yes", "No"], n_samples),
        "Dependents": np.random.choice(["Yes", "No"], n_samples, p=[0.3, 0.7]),
        "tenure": np.random.randint(0, 73, n_samples),
        "PhoneService": np.random.choice(["Yes", "No"], n_samples, p=[0.9, 0.1]),
        "MultipleLines": np.random.choice(
            ["Yes", "No", "No phone service"], n_samples, p=[0.42, 0.48, 0.1]
        ),
        "InternetService": np.random.choice(
            ["DSL", "Fiber optic", "No"], n_samples, p=[0.34, 0.44, 0.22]
        ),
        "OnlineSecurity": np.random.choice(
            ["Yes", "No", "No internet service"], n_samples, p=[0.29, 0.49, 0.22]
        ),
        "OnlineBackup": np.random.choice(
            ["Yes", "No", "No internet service"], n_samples, p=[0.34, 0.44, 0.22]
        ),
        "DeviceProtection": np.random.choice(
            ["Yes", "No", "No internet service"], n_samples, p=[0.34, 0.44, 0.22]
        ),
        "TechSupport": np.random.choice(
            ["Yes", "No", "No internet service"], n_samples, p=[0.29, 0.49, 0.22]
        ),
        "StreamingTV": np.random.choice(
            ["Yes", "No", "No internet service"], n_samples, p=[0.38, 0.40, 0.22]
        ),
        "StreamingMovies": np.random.choice(
            ["Yes", "No", "No internet service"], n_samples, p=[0.38, 0.40, 0.22]
        ),
        "Contract": np.random.choice(
            ["Month-to-month", "One year", "Two year"], n_samples, p=[0.55, 0.21, 0.24]
        ),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n_samples, p=[0.59, 0.41]),
        "PaymentMethod": np.random.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            n_samples,
            p=[0.34, 0.22, 0.22, 0.22],
        ),
        "MonthlyCharges": np.round(np.random.uniform(18, 119, n_samples), 2),
    }
    
    # Calculate TotalCharges based on tenure and MonthlyCharges
    data["TotalCharges"] = np.round(
        data["tenure"] * data["MonthlyCharges"] + np.random.uniform(-50, 50, n_samples), 2
    )
    data["TotalCharges"] = np.maximum(data["TotalCharges"], 0)
    
    # Generate Churn based on factors (higher churn for month-to-month, short tenure, etc.)
    churn_prob = np.zeros(n_samples)
    
    # Month-to-month contracts have higher churn
    churn_prob += np.where(np.array(data["Contract"]) == "Month-to-month", 0.3, 0)
    
    # Short tenure increases churn probability
    churn_prob += np.where(np.array(data["tenure"]) < 12, 0.2, 0)
    
    # Higher monthly charges increase churn
    churn_prob += np.where(np.array(data["MonthlyCharges"]) > 70, 0.15, 0)
    
    # Electronic check payment increases churn
    churn_prob += np.where(np.array(data["PaymentMethod"]) == "Electronic check", 0.1, 0)
    
    # No tech support increases churn
    churn_prob += np.where(np.array(data["TechSupport"]) == "No", 0.05, 0)
    
    # Normalize probability
    churn_prob = np.clip(churn_prob, 0, 0.9)
    
    # Generate churn
    data["Churn"] = np.where(
        np.random.random(n_samples) < churn_prob, "Yes", "No"
    )
    
    df = pd.DataFrame(data)
    
    # Save sample data
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_FILE, index=False)
    logger.info(f"Sample data saved to {RAW_DATA_FILE}")
    
    return df


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.
    
    Args:
        df: DataFrame to analyze.
        
    Returns:
        Dictionary containing data information.
    """
    info = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
    }
    
    return info


if __name__ == "__main__":
    # Test data loading
    df = load_raw_data()
    info = get_data_info(df)
    print(f"Data shape: {info['n_rows']} rows, {info['n_columns']} columns")
    print(f"Missing values: {sum(info['missing_values'].values())}")
