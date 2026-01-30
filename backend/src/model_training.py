"""
Machine Learning Model Training Pipeline for Customer Churn Prediction.
Implements multiple models with cross-validation and hyperparameter tuning.
"""
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    RANDOM_STATE,
    CV_FOLDS,
    MODELS_DIR,
    BEST_MODEL_PATH,
    MODEL_METRICS_PATH,
    FEATURE_IMPORTANCE_PATH,
    LOGISTIC_REGRESSION_PARAMS,
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
)


class ChurnModelTrainer:
    """
    Model trainer for customer churn prediction.
    Supports multiple model types with cross-validation and hyperparameter tuning.
    """
    
    def __init__(self, random_state: int = RANDOM_STATE, cv_folds: int = CV_FOLDS):
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        self.feature_importance = None
        
    def get_base_models(self) -> Dict[str, Any]:
        """
        Get dictionary of base models to train.
        
        Returns:
            Dictionary of model name to model instance.
        """
        models = {
            "Logistic Regression": LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            "Random Forest": RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1
            ),
            "XGBoost": XGBClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric="logloss"
            )
        }
        return models
    
    def get_param_grids(self) -> Dict[str, Dict]:
        """
        Get hyperparameter grids for each model.
        
        Returns:
            Dictionary of model name to parameter grid.
        """
        return {
            "Logistic Regression": LOGISTIC_REGRESSION_PARAMS,
            "Random Forest": RANDOM_FOREST_PARAMS,
            "XGBoost": XGBOOST_PARAMS
        }
    
    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained model.
            X_test: Test features.
            y_test: Test labels.
            model_name: Name of the model.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics["classification_report"] = report
        
        logger.info(f"\n{model_name} Evaluation Results:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def cross_validate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        scoring: str = "roc_auc"
    ) -> Dict[str, float]:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to validate.
            X: Features.
            y: Labels.
            model_name: Name of the model.
            scoring: Scoring metric.
            
        Returns:
            Dictionary of cross-validation results.
        """
        logger.info(f"Cross-validating {model_name}...")
        
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Multiple scoring metrics
        scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        cv_results = {}
        
        for metric in scoring_metrics:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
            cv_results[f"{metric}_mean"] = scores.mean()
            cv_results[f"{metric}_std"] = scores.std()
        
        logger.info(f"  {model_name} CV {scoring}: {cv_results[f'{scoring}_mean']:.4f} (+/- {cv_results[f'{scoring}_std']:.4f})")
        
        return cv_results
    
    def tune_hyperparameters(
        self,
        model: Any,
        param_grid: Dict,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        scoring: str = "roc_auc"
    ) -> Tuple[Any, Dict]:
        """
        Tune hyperparameters using GridSearchCV.
        
        Args:
            model: Model to tune.
            param_grid: Parameter grid.
            X: Features.
            y: Labels.
            model_name: Name of the model.
            scoring: Scoring metric.
            
        Returns:
            Tuple of (best model, best parameters).
        """
        logger.info(f"Tuning hyperparameters for {model_name}...")
        
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=1,
            refit=True
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"  Best {scoring}: {grid_search.best_score_:.4f}")
        logger.info(f"  Best params: {grid_search.best_params_}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def handle_class_imbalance(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance using SMOTE.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            
        Returns:
            Resampled X and y.
        """
        logger.info("Handling class imbalance with SMOTE...")
        logger.info(f"  Original distribution: {np.bincount(y_train)}")
        
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        logger.info(f"  Resampled distribution: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        use_smote: bool = True,
        tune_hyperparameters: bool = True
    ) -> Dict[str, Dict]:
        """
        Train all models and select the best one.
        
        Args:
            X_train: Training features.
            X_test: Test features.
            y_train: Training labels.
            y_test: Test labels.
            use_smote: Whether to use SMOTE for class imbalance.
            tune_hyperparameters: Whether to tune hyperparameters.
            
        Returns:
            Dictionary of results for all models.
        """
        # Handle class imbalance
        if use_smote:
            X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Get models and param grids
        base_models = self.get_base_models()
        param_grids = self.get_param_grids()
        
        results = {}
        best_score = 0
        
        for model_name, model in base_models.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_name}")
            logger.info(f"{'='*50}")
            
            # Hyperparameter tuning
            if tune_hyperparameters and model_name in param_grids:
                # Use smaller param grid for faster training
                small_param_grid = self._get_small_param_grid(param_grids[model_name])
                model, best_params = self.tune_hyperparameters(
                    model, small_param_grid, X_train_balanced, y_train_balanced, model_name
                )
            else:
                model.fit(X_train_balanced, y_train_balanced)
                best_params = {}
            
            # Cross-validation
            cv_results = self.cross_validate_model(
                model, X_train_balanced, y_train_balanced, model_name
            )
            
            # Evaluate on test set
            test_metrics = self.evaluate_model(model, X_test, y_test, model_name)
            
            # Store results
            results[model_name] = {
                "model": model,
                "cv_results": cv_results,
                "test_metrics": test_metrics,
                "best_params": best_params
            }
            
            # Store model
            self.models[model_name] = model
            
            # Check if best model
            if test_metrics["roc_auc"] > best_score:
                best_score = test_metrics["roc_auc"]
                self.best_model = model
                self.best_model_name = model_name
        
        self.results = results
        
        logger.info(f"\n{'='*50}")
        logger.info(f"BEST MODEL: {self.best_model_name}")
        logger.info(f"Best ROC-AUC: {best_score:.4f}")
        logger.info(f"{'='*50}")
        
        return results
    
    def _get_small_param_grid(self, param_grid: Dict) -> Dict:
        """Get a smaller parameter grid for faster tuning."""
        small_grid = {}
        for key, values in param_grid.items():
            if isinstance(values, list) and len(values) > 2:
                # Take first, middle, and last values
                small_grid[key] = [values[0], values[len(values)//2], values[-1]]
            else:
                small_grid[key] = values
        return small_grid
    
    def get_feature_importance(
        self,
        feature_names: List[str],
        model: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        Get feature importance from the model.
        
        Args:
            feature_names: List of feature names.
            model: Model to get importance from (default: best model).
            
        Returns:
            DataFrame with feature importance.
        """
        if model is None:
            model = self.best_model
        
        if model is None:
            raise ValueError("No model available. Train models first.")
        
        # Get feature importance
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_[0])
        else:
            logger.warning("Model doesn't support feature importance")
            return pd.DataFrame()
        
        # Ensure lengths match
        if len(importance) != len(feature_names):
            logger.warning(f"Feature importance length ({len(importance)}) != feature names length ({len(feature_names)})")
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)
        
        self.feature_importance = importance_df
        
        return importance_df
    
    def save_model(
        self,
        filepath: Path = BEST_MODEL_PATH,
        model: Optional[Any] = None
    ):
        """Save model to disk."""
        if model is None:
            model = self.best_model
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Path = BEST_MODEL_PATH) -> Any:
        """Load model from disk."""
        if not filepath.exists():
            raise FileNotFoundError(f"Model not found at {filepath}")
        
        self.best_model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return self.best_model
    
    def save_results(
        self,
        metrics_path: Path = MODEL_METRICS_PATH,
        importance_path: Path = FEATURE_IMPORTANCE_PATH
    ):
        """Save training results and metrics."""
        # Save metrics
        metrics_to_save = {}
        for model_name, result in self.results.items():
            metrics_to_save[model_name] = {
                "cv_results": {k: float(v) if isinstance(v, (np.floating, float)) else v 
                              for k, v in result["cv_results"].items()},
                "test_metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v 
                                for k, v in result["test_metrics"].items()
                                if k not in ["confusion_matrix", "classification_report"]},
                "best_params": {k: int(v) if isinstance(v, np.integer) else v 
                               for k, v in result["best_params"].items()}
            }
        
        metrics_to_save["best_model"] = self.best_model_name
        
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics_to_save, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Save feature importance
        if self.feature_importance is not None:
            importance_dict = self.feature_importance.to_dict(orient="records")
            with open(importance_path, "w") as f:
                json.dump(importance_dict, f, indent=2)
            logger.info(f"Feature importance saved to {importance_path}")


def train_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    use_smote: bool = True,
    tune_hyperparameters: bool = True
) -> Tuple[Any, Dict, pd.DataFrame]:
    """
    Main function to train all models.
    
    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training labels.
        y_test: Test labels.
        feature_names: List of feature names.
        use_smote: Whether to use SMOTE.
        tune_hyperparameters: Whether to tune hyperparameters.
        
    Returns:
        Tuple of (best model, results dictionary, feature importance DataFrame).
    """
    trainer = ChurnModelTrainer()
    
    # Train all models
    results = trainer.train_all_models(
        X_train, X_test, y_train, y_test,
        use_smote=use_smote,
        tune_hyperparameters=tune_hyperparameters
    )
    
    # Get feature importance
    importance_df = trainer.get_feature_importance(feature_names)
    
    # Save model and results
    trainer.save_model()
    trainer.save_results()
    
    return trainer.best_model, results, importance_df


if __name__ == "__main__":
    from data_loader import load_raw_data
    from preprocessing import DataPreprocessor, split_data
    
    # Load and preprocess data
    logger.info("Loading data...")
    df = load_raw_data()
    
    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    df_engineered = preprocessor.engineer_features(df_clean)
    
    X, y = preprocessor.prepare_data(df_engineered, fit=True)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    # Train models
    logger.info("Training models...")
    feature_names = preprocessor.get_feature_names()
    best_model, results, importance_df = train_models(
        X_train, X_test, y_train, y_test, feature_names,
        use_smote=True,
        tune_hyperparameters=True
    )
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("TRAINING COMPLETE")
    logger.info("="*50)
    logger.info(f"\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
