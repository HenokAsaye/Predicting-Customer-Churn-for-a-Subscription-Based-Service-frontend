"""
Data preprocessing module for Customer Churn Prediction.
Handles data cleaning, feature engineering, and transformation.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    PROCESSED_DATA_DIR,
    PREPROCESSOR_PATH,
)


class DataPreprocessor:
    """
    Data preprocessing class for customer churn prediction.
    Handles cleaning, feature engineering, and transformations.
    """
    
    def __init__(self):
        self.preprocessor = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw data by handling missing values and data type issues.
        
        Args:
            df: Raw DataFrame.
            
        Returns:
            Cleaned DataFrame.
        """
        logger.info("Starting data cleaning...")
        df = df.copy()
        
        try:
            # Drop customerID as it's not a feature
            if "customerID" in df.columns:
                df = df.drop("customerID", axis=1)
                logger.info("Dropped customerID column")
            
            # Handle TotalCharges - convert to numeric and handle empty strings
            if "TotalCharges" in df.columns:
                # Replace empty strings with NaN
                df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
                
                # Fill missing TotalCharges with MonthlyCharges * tenure
                missing_total = df["TotalCharges"].isnull()
                if missing_total.any():
                    df.loc[missing_total, "TotalCharges"] = (
                        df.loc[missing_total, "MonthlyCharges"] * 
                        df.loc[missing_total, "tenure"]
                    )
                    logger.info(f"Imputed {missing_total.sum()} missing TotalCharges values")
            
            # Ensure SeniorCitizen is string for categorical encoding
            if "SeniorCitizen" in df.columns:
                if df["SeniorCitizen"].dtype in ['int', 'int64', 'int32']:
                    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
                else:
                    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
                    df["SeniorCitizen"] = df["SeniorCitizen"].map({"0": "No", "1": "Yes", "No": "No", "Yes": "Yes"})
            
            # Handle any remaining missing values in categorical columns
            for col in CATEGORICAL_FEATURES:
                if col in df.columns and df[col].isnull().any():
                    mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "No"
                    df[col] = df[col].fillna(mode_val)
                    logger.info(f"Filled missing values in {col} with mode: {mode_val}")
            
            # Handle any remaining missing values in numeric columns
            for col in NUMERIC_FEATURES:
                if col in df.columns and df[col].isnull().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.info(f"Filled missing values in {col} with median: {median_val}")
            
            # Remove duplicates
            n_duplicates = df.duplicated().sum()
            if n_duplicates > 0:
                df = df.drop_duplicates()
                logger.info(f"Removed {n_duplicates} duplicate rows")
            
            logger.info(f"Data cleaning complete. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error in clean_data: {e}", exc_info=True)
            raise
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones.
        
        Args:
            df: Cleaned DataFrame.
            
        Returns:
            DataFrame with engineered features.
        """
        logger.info("Starting feature engineering...")
        df = df.copy()
        
        try:
            # Tenure groups
            df["TenureGroup"] = pd.cut(
                df["tenure"],
                bins=[0, 12, 24, 48, 60, 72],
                labels=["0-12", "12-24", "24-48", "48-60", "60-72"],
                include_lowest=True
            ).astype(str)
            
            # Average monthly spend (Total / tenure, handle division by zero)
            df["AvgMonthlySpend"] = np.where(
                df["tenure"] > 0,
                df["TotalCharges"] / df["tenure"],
                df["MonthlyCharges"]
            )
            
            # Charge increase indicator (if current monthly is higher than average)
            df["ChargeIncrease"] = np.where(
                df["MonthlyCharges"] > df["AvgMonthlySpend"],
                "Yes",
                "No"
            )
            
            # Number of services subscribed
            service_cols = [
                "PhoneService", "MultipleLines", "InternetService",
                "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"
            ]
            
            df["NumServices"] = 0
            for col in service_cols:
                if col in df.columns:
                    df["NumServices"] += (df[col].isin(["Yes", "DSL", "Fiber optic"])).astype(int)
            
            # Has partner and dependents
            if "Partner" in df.columns and "Dependents" in df.columns:
                df["HasFamily"] = np.where(
                    (df["Partner"] == "Yes") | (df["Dependents"] == "Yes"),
                    "Yes",
                    "No"
                )
            
            # Contract security (long contract + online security)
            if "Contract" in df.columns and "OnlineSecurity" in df.columns:
                df["SecureContract"] = np.where(
                    (df["Contract"] != "Month-to-month") & (df["OnlineSecurity"] == "Yes"),
                    "Yes",
                    "No"
                )
            
            logger.info(f"Feature engineering complete. New shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}", exc_info=True)
            raise
    
    def build_preprocessor(
        self,
        numeric_features: List[str],
        categorical_features: List[str]
    ) -> ColumnTransformer:
        """
        Build a preprocessing pipeline for numeric and categorical features.
        
        Args:
            numeric_features: List of numeric feature names.
            categorical_features: List of categorical feature names.
            
        Returns:
            ColumnTransformer for preprocessing.
        """
        # Numeric pipeline: impute then scale
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        
        # Categorical pipeline: impute then one-hot encode
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        
        # Combine pipelines
        preprocessor = ColumnTransformer([
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features)
        ])
        
        return preprocessor
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        fit: bool = True,
        include_engineered: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare data for model training or prediction.
        
        Args:
            df: DataFrame to prepare.
            fit: Whether to fit the preprocessor (True for training).
            include_engineered: Whether to include engineered features.
            
        Returns:
            Tuple of (X transformed, y if target exists).
        """
        df = df.copy()
        
        # Define features to use
        numeric_features = NUMERIC_FEATURES.copy()
        categorical_features = CATEGORICAL_FEATURES.copy()
        
        if include_engineered:
            # Add engineered numeric features
            if "AvgMonthlySpend" in df.columns:
                numeric_features.append("AvgMonthlySpend")
            if "NumServices" in df.columns:
                numeric_features.append("NumServices")
            
            # Add engineered categorical features
            for col in ["TenureGroup", "ChargeIncrease", "HasFamily", "SecureContract"]:
                if col in df.columns:
                    categorical_features.append(col)
        
        # Filter to only existing columns
        numeric_features = [f for f in numeric_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]
        
        logger.debug(f"Using numeric features: {numeric_features}")
        logger.debug(f"Using categorical features: {categorical_features}")
        logger.debug(f"DataFrame shape before selection: {df.shape}")
        logger.debug(f"DataFrame columns: {df.columns.tolist()}")
        
        # Extract features
        all_features = numeric_features + categorical_features
        missing_features = [f for f in all_features if f not in df.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        X = df[all_features]
        logger.debug(f"X shape after feature selection: {X.shape}")
        
        # Extract target if exists
        y = None
        if TARGET_COLUMN in df.columns:
            y = (df[TARGET_COLUMN] == "Yes").astype(int).values
        
        # Build or use existing preprocessor
        if fit:
            self.preprocessor = self.build_preprocessor(
                numeric_features, categorical_features
            )
            X_transformed = self.preprocessor.fit_transform(X)
            
            # Store feature names
            self._store_feature_names(numeric_features, categorical_features)
            logger.info(f"Fitted preprocessor. Output shape: {X_transformed.shape}")
        else:
            if self.preprocessor is None:
                raise ValueError("Preprocessor not fitted. Call prepare_data with fit=True first.")
            X_transformed = self.preprocessor.transform(X)
            logger.debug(f"Transformed data shape: {X_transformed.shape}")
        
        return X_transformed, y
    
    def _store_feature_names(
        self,
        numeric_features: List[str],
        categorical_features: List[str]
    ):
        """Store feature names after transformation."""
        self.feature_names = numeric_features.copy()
        
        # Get one-hot encoded feature names
        if hasattr(self.preprocessor, "named_transformers_"):
            cat_encoder = self.preprocessor.named_transformers_["categorical"]
            if hasattr(cat_encoder, "named_steps") and "encoder" in cat_encoder.named_steps:
                encoder = cat_encoder.named_steps["encoder"]
                if hasattr(encoder, "get_feature_names_out"):
                    cat_feature_names = encoder.get_feature_names_out(categorical_features)
                    self.feature_names.extend(cat_feature_names)
    
    def get_feature_names(self) -> List[str]:
        """Get the names of transformed features."""
        return self.feature_names if self.feature_names else []
    
    def save_preprocessor(self, filepath: Path = PREPROCESSOR_PATH):
        """Save the preprocessor to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "preprocessor": self.preprocessor,
            "feature_names": self.feature_names
        }, filepath)
        logger.info(f"Saved preprocessor to {filepath}")
    
    def load_preprocessor(self, filepath: Path = PREPROCESSOR_PATH):
        """Load the preprocessor from disk."""
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor not found at {filepath}")
        
        data = joblib.load(filepath)
        self.preprocessor = data["preprocessor"]
        self.feature_names = data["feature_names"]
        logger.info(f"Loaded preprocessor from {filepath}")


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Proportion of data for testing.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    logger.info(f"Churn rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from data_loader import load_raw_data
    
    # Load and preprocess data
    df = load_raw_data()
    
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    df_engineered = preprocessor.engineer_features(df_clean)
    
    X, y = preprocessor.prepare_data(df_engineered, fit=True)
    print(f"Prepared data shape: X={X.shape}, y={y.shape}")
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Split complete: Train={X_train.shape}, Test={X_test.shape}")
    
    # Save preprocessor
    preprocessor.save_preprocessor()
