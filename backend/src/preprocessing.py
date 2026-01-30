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
    def __init__(self):
        self.preprocessor = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Initializing raw data cleaning process")
        df = df.copy()

        try:
            if "customerID" in df.columns:
                df = df.drop("customerID", axis=1)
                logger.info("Removed non-informative identifier column: customerID")

            if "TotalCharges" in df.columns:
                df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
                missing_total = df["TotalCharges"].isnull()

                if missing_total.any():
                    df.loc[missing_total, "TotalCharges"] = (
                        df.loc[missing_total, "MonthlyCharges"]
                        * df.loc[missing_total, "tenure"]
                    )
                    logger.info(
                        f"Computed TotalCharges for {missing_total.sum()} records using MonthlyCharges and tenure"
                    )

            if "SeniorCitizen" in df.columns:
                if df["SeniorCitizen"].dtype in ["int", "int64", "int32"]:
                    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
                else:
                    df["SeniorCitizen"] = (
                        df["SeniorCitizen"]
                        .astype(str)
                        .map({"0": "No", "1": "Yes", "No": "No", "Yes": "Yes"})
                    )
                logger.info("Normalized SeniorCitizen values to categorical format")

            for col in CATEGORICAL_FEATURES:
                if col in df.columns and df[col].isnull().any():
                    mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "No"
                    df[col] = df[col].fillna(mode_val)
                    logger.info(
                        f"Filled missing categorical values in '{col}' using mode='{mode_val}'"
                    )

            for col in NUMERIC_FEATURES:
                if col in df.columns and df[col].isnull().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.info(
                        f"Filled missing numeric values in '{col}' using median={median_val}"
                    )

            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                df = df.drop_duplicates()
                logger.info(f"Removed {duplicate_count} duplicate records")

            logger.info(f"Data cleaning completed successfully | Final shape: {df.shape}")
            return df

        except Exception as e:
            logger.error("Data cleaning failed due to an unexpected error", exc_info=True)
            raise

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting feature engineering pipeline")
        df = df.copy()

        try:
            df["TenureGroup"] = pd.cut(
                df["tenure"],
                bins=[0, 12, 24, 48, 60, 72],
                labels=["0-12", "12-24", "24-48", "48-60", "60-72"],
                include_lowest=True,
            ).astype(str)
            

            df["AvgMonthlySpend"] = np.where(
                df["tenure"] > 0,
                df["TotalCharges"] / df["tenure"],
                df["MonthlyCharges"],
            )

            df["ChargeIncrease"] = np.where(
                df["MonthlyCharges"] > df["AvgMonthlySpend"], "Yes", "No"
            )

            service_cols = [
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
            ]

            df["NumServices"] = 0
            for col in service_cols:
                if col in df.columns:
                    df["NumServices"] += (
                        df[col].isin(["Yes", "DSL", "Fiber optic"])
                    ).astype(int)

            if "Partner" in df.columns and "Dependents" in df.columns:
                df["HasFamily"] = np.where(
                    (df["Partner"] == "Yes") | (df["Dependents"] == "Yes"),
                    "Yes",
                    "No",
                )

            if "Contract" in df.columns and "OnlineSecurity" in df.columns:
                df["SecureContract"] = np.where(
                    (df["Contract"] != "Month-to-month")
                    & (df["OnlineSecurity"] == "Yes"),
                    "Yes",
                    "No",
                )

            logger.info(
                f"Feature engineering completed | Total features now: {df.shape[1]}"
            )
            return df

        except Exception as e:
            logger.error(
                "Feature engineering failed due to an unexpected error", exc_info=True
            )
            raise

    def build_preprocessor(
        self, numeric_features: List[str], categorical_features: List[str]
    ) -> ColumnTransformer:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        return ColumnTransformer(
            [
                ("numeric", numeric_pipeline, numeric_features),
                ("categorical", categorical_pipeline, categorical_features),
            ]
        )

    def prepare_data(
        self,
        df: pd.DataFrame,
        fit: bool = True,
        include_engineered: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        df = df.copy()

        numeric_features = NUMERIC_FEATURES.copy()
        categorical_features = CATEGORICAL_FEATURES.copy()

        if include_engineered:
            if "AvgMonthlySpend" in df.columns:
                numeric_features.append("AvgMonthlySpend")
            if "NumServices" in df.columns:
                numeric_features.append("NumServices")

            for col in [
                "TenureGroup",
                "ChargeIncrease",
                "HasFamily",
                "SecureContract",
            ]:
                if col in df.columns:
                    categorical_features.append(col)

        numeric_features = [f for f in numeric_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]

        logger.debug(f"Selected numeric features: {numeric_features}")
        logger.debug(f"Selected categorical features: {categorical_features}")

        all_features = numeric_features + categorical_features
        X = df[all_features]

        y = None
        if TARGET_COLUMN in df.columns:
            y = (df[TARGET_COLUMN] == "Yes").astype(int).values
            logger.info("Target variable extracted and encoded")

        

        if fit:
            self.preprocessor = self.build_preprocessor(
                numeric_features, categorical_features
            )
            X_transformed = self.preprocessor.fit_transform(X)
            self._store_feature_names(numeric_features, categorical_features)
            logger.info(
                f"Preprocessor fitted successfully | Output shape: {X_transformed.shape}"
            )
        else:
            if self.preprocessor is None:
                raise ValueError(
                    "Preprocessor has not been fitted. Run with fit=True first."
                )
            X_transformed = self.preprocessor.transform(X)
            logger.debug(
                f"Data transformed using existing preprocessor | Shape: {X_transformed.shape}"
            )

        return X_transformed, y

    def _store_feature_names(
        self, numeric_features: List[str], categorical_features: List[str]
    ):
        self.feature_names = numeric_features.copy()

        if hasattr(self.preprocessor, "named_transformers_"):
            cat_encoder = self.preprocessor.named_transformers_["categorical"]
            if "encoder" in cat_encoder.named_steps:
                encoder = cat_encoder.named_steps["encoder"]
                if hasattr(encoder, "get_feature_names_out"):
                    self.feature_names.extend(
                        encoder.get_feature_names_out(categorical_features)
                    )

    def get_feature_names(self) -> List[str]:
        return self.feature_names if self.feature_names else []

    def save_preprocessor(self, filepath: Path = PREPROCESSOR_PATH):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"preprocessor": self.preprocessor, "feature_names": self.feature_names},
            filepath,
        )
        logger.info(f"Preprocessor persisted successfully at: {filepath}")

    def load_preprocessor(self, filepath: Path = PREPROCESSOR_PATH):
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor file not found at: {filepath}")

        data = joblib.load(filepath)
        self.preprocessor = data["preprocessor"]
        self.feature_names = data["feature_names"]
        logger.info(f"Preprocessor loaded successfully from: {filepath}")


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(f"Training samples: {X_train.shape[0]}")
    logger.info(f"Testing samples: {X_test.shape[0]}")
    logger.info(
        f"Churn distribution | Train: {y_train.mean():.2%} | Test: {y_test.mean():.2%}"
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from data_loader import load_raw_data

    df = load_raw_data()

    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    df_engineered = preprocessor.engineer_features(df_clean)

    X, y = preprocessor.prepare_data(df_engineered, fit=True)
    print(f"Prepared data shape: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Split complete: Train={X_train.shape}, Test={X_test.shape}")

    preprocessor.save_preprocessor()
