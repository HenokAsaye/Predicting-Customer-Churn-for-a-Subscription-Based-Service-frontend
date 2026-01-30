"""
Model evaluation utilities for Customer Churn Prediction.
Provides comprehensive evaluation metrics and visualizations.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import FIGURES_DIR, MODELS_DIR


class ModelEvaluator:
    """
    Comprehensive model evaluation class.
    """
    
    def __init__(self, model: Any, model_name: str = "Model"):
        self.model = model
        self.model_name = model_name
        self.metrics = {}
        
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features.
            y_test: True labels.
            threshold: Classification threshold.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        # Get predictions
        y_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "average_precision": average_precision_score(y_test, y_proba),
            "threshold": threshold
        }
        
        # Confusion matrix
        self.metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred)
        
        # Store predictions for later use
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_proba = y_proba
        
        return self.metrics
    
    def print_report(self):
        """Print evaluation report."""
        print(f"\n{'='*50}")
        print(f"Evaluation Report: {self.model_name}")
        print(f"{'='*50}")
        print(f"Accuracy:          {self.metrics['accuracy']:.4f}")
        print(f"Precision:         {self.metrics['precision']:.4f}")
        print(f"Recall:            {self.metrics['recall']:.4f}")
        print(f"F1-Score:          {self.metrics['f1_score']:.4f}")
        print(f"ROC-AUC:           {self.metrics['roc_auc']:.4f}")
        print(f"Average Precision: {self.metrics['average_precision']:.4f}")
        print(f"\nConfusion Matrix:")
        print(self.metrics['confusion_matrix'])
        print(f"\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred, 
              target_names=['Not Churn', 'Churn']))
    
    def plot_confusion_matrix(
        self,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """Plot confusion matrix heatmap."""
        plt.figure(figsize=figsize)
        
        cm = self.metrics['confusion_matrix']
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Churn', 'Churn'],
            yticklabels=['Not Churn', 'Churn']
        )
        
        plt.title(f'Confusion Matrix - {self.model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(
        self,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """Plot ROC curve."""
        plt.figure(figsize=figsize)
        
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_proba)
        roc_auc = self.metrics['roc_auc']
        
        plt.plot(fpr, tpr, color='#3498db', lw=2, 
                label=f'{self.model_name} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {self.model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(
        self,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """Plot Precision-Recall curve."""
        plt.figure(figsize=figsize)
        
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_proba)
        avg_precision = self.metrics['average_precision']
        
        plt.plot(recall, precision, color='#e74c3c', lw=2,
                label=f'{self.model_name} (AP = {avg_precision:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {self.model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve saved to {save_path}")
        
        plt.show()
    
    def plot_threshold_analysis(
        self,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """Plot metrics vs threshold analysis."""
        plt.figure(figsize=figsize)
        
        thresholds = np.arange(0.1, 1.0, 0.05)
        metrics_by_threshold = {
            'precision': [],
            'recall': [],
            'f1_score': [],
            'accuracy': []
        }
        
        for thresh in thresholds:
            y_pred_thresh = (self.y_proba >= thresh).astype(int)
            metrics_by_threshold['precision'].append(
                precision_score(self.y_test, y_pred_thresh, zero_division=0))
            metrics_by_threshold['recall'].append(
                recall_score(self.y_test, y_pred_thresh, zero_division=0))
            metrics_by_threshold['f1_score'].append(
                f1_score(self.y_test, y_pred_thresh, zero_division=0))
            metrics_by_threshold['accuracy'].append(
                accuracy_score(self.y_test, y_pred_thresh))
        
        plt.plot(thresholds, metrics_by_threshold['precision'], 
                label='Precision', color='#3498db', lw=2)
        plt.plot(thresholds, metrics_by_threshold['recall'], 
                label='Recall', color='#e74c3c', lw=2)
        plt.plot(thresholds, metrics_by_threshold['f1_score'], 
                label='F1-Score', color='#2ecc71', lw=2)
        plt.plot(thresholds, metrics_by_threshold['accuracy'], 
                label='Accuracy', color='#9b59b6', lw=2)
        
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title(f'Metrics vs Threshold - {self.model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Find optimal threshold (max F1)
        optimal_idx = np.argmax(metrics_by_threshold['f1_score'])
        optimal_threshold = thresholds[optimal_idx]
        plt.axvline(x=optimal_threshold, color='gray', linestyle='--', 
                   label=f'Optimal Threshold: {optimal_threshold:.2f}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Threshold analysis saved to {save_path}")
        
        plt.show()
        
        return optimal_threshold
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        top_n: int = 15,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """Plot feature importance."""
        plt.figure(figsize=figsize)
        
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            logger.warning("Model doesn't support feature importance")
            return
        
        # Ensure lengths match
        if len(importance) != len(feature_names):
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True).tail(top_n)
        
        # Plot
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, top_n))
        plt.barh(importance_df['feature'], importance_df['importance'], color=colors)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance - {self.model_name}', 
                 fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance saved to {save_path}")
        
        plt.show()
    
    def generate_full_report(
        self,
        feature_names: List[str],
        save_dir: Path = FIGURES_DIR
    ):
        """Generate full evaluation report with all plots."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Print text report
        self.print_report()
        
        # Generate all plots
        self.plot_confusion_matrix(save_dir / f'{self.model_name}_confusion_matrix.png')
        self.plot_roc_curve(save_dir / f'{self.model_name}_roc_curve.png')
        self.plot_precision_recall_curve(save_dir / f'{self.model_name}_pr_curve.png')
        optimal_threshold = self.plot_threshold_analysis(
            save_dir / f'{self.model_name}_threshold_analysis.png')
        self.plot_feature_importance(
            feature_names, save_path=save_dir / f'{self.model_name}_feature_importance.png')
        
        logger.info(f"\nOptimal threshold: {optimal_threshold:.2f}")
        logger.info(f"All plots saved to {save_dir}")


def compare_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> pd.DataFrame:
    """
    Compare multiple models.
    
    Args:
        models: Dictionary of model name to model.
        X_test: Test features.
        y_test: Test labels.
        save_path: Path to save comparison plot.
        figsize: Figure size.
        
    Returns:
        DataFrame with comparison results.
    """
    results = []
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ROC curves
    for model_name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Metrics
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_proba)
        }
        results.append(metrics)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        axes[0].plot(fpr, tpr, lw=2, label=f"{model_name} (AUC={metrics['ROC-AUC']:.3f})")
        
        # PR curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        axes[1].plot(recall, precision, lw=2, label=f"{model_name}")
    
    # ROC plot formatting
    axes[0].plot([0, 1], [0, 1], 'k--', lw=2)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curves Comparison', fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # PR plot formatting
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curves Comparison', fontweight='bold')
    axes[1].legend(loc='lower left')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
    
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    
    return comparison_df


if __name__ == "__main__":
    import joblib
    from backend.config import BEST_MODEL_PATH, PREPROCESSOR_PATH
    from data_loader import load_raw_data
    from preprocessing import DataPreprocessor, split_data
    
    # Load data and preprocess
    df = load_raw_data()
    preprocessor = DataPreprocessor()
    preprocessor.load_preprocessor()
    
    df_clean = preprocessor.clean_data(df)
    df_engineered = preprocessor.engineer_features(df_clean)
    X, y = preprocessor.prepare_data(df_engineered, fit=False)
    
    _, X_test, _, y_test = split_data(X, y)
    
    # Load model
    model = joblib.load(BEST_MODEL_PATH)
    
    # Evaluate
    evaluator = ModelEvaluator(model, "Best Model")
    evaluator.evaluate(X_test, y_test)
    evaluator.generate_full_report(preprocessor.get_feature_names())
