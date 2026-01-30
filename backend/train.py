"""
Main training script for Customer Churn Prediction.
Run this script to train all models and select the best one.
"""
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import load_raw_data
from src.preprocessing import DataPreprocessor, split_data
from src.model_training import train_models
from src.evaluation import ModelEvaluator
from backend.config import FIGURES_DIR


def main():
    """Main function to run the entire training pipeline."""
    logger.info("="*60)
    logger.info("CUSTOMER CHURN PREDICTION - MODEL TRAINING PIPELINE")
    logger.info("="*60)
    
    # Step 1: Load data
    logger.info("\n📊 Step 1: Loading data...")
    df = load_raw_data()
    logger.info(f"Loaded {len(df)} records")
    
    # Step 2: Preprocess data
    logger.info("\n🔧 Step 2: Preprocessing data...")
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    df_engineered = preprocessor.engineer_features(df_clean)
    
    # Step 3: Prepare features
    logger.info("\n📐 Step 3: Preparing features...")
    X, y = preprocessor.prepare_data(df_engineered, fit=True)
    logger.info(f"Feature matrix shape: {X.shape}")
    
    # Step 4: Split data
    logger.info("\n✂️ Step 4: Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 5: Save preprocessor
    logger.info("\n💾 Step 5: Saving preprocessor...")
    preprocessor.save_preprocessor()
    
    # Step 6: Train models
    logger.info("\n🤖 Step 6: Training models...")
    feature_names = preprocessor.get_feature_names()
    best_model, results, importance_df = train_models(
        X_train, X_test, y_train, y_test, feature_names,
        use_smote=True,
        tune_hyperparameters=True
    )
    
    # Step 7: Evaluate best model
    logger.info("\n📈 Step 7: Evaluating best model...")
    evaluator = ModelEvaluator(best_model, "Best Model")
    evaluator.evaluate(X_test, y_test)
    evaluator.generate_full_report(feature_names, FIGURES_DIR)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("="*60)
    
    logger.info("\n📊 Model Performance Summary:")
    for model_name, result in results.items():
        metrics = result["test_metrics"]
        logger.info(f"\n{model_name}:")
        logger.info(f"  - Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  - Precision: {metrics['precision']:.4f}")
        logger.info(f"  - Recall:    {metrics['recall']:.4f}")
        logger.info(f"  - F1-Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  - ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    logger.info("\n🏆 Top 10 Important Features:")
    for i, row in importance_df.head(10).iterrows():
        logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    logger.info("\n📁 Output Files:")
    logger.info("  - Model: models/best_model.joblib")
    logger.info("  - Preprocessor: models/preprocessor.joblib")
    logger.info("  - Metrics: models/model_metrics.json")
    logger.info("  - Features: models/feature_importance.json")
    logger.info(f"  - Figures: {FIGURES_DIR}/")
    
    logger.info("\n🚀 Next Steps:")
    logger.info("  1. Start the API: uvicorn api.main:app --reload")
    logger.info("  2. Start Streamlit: streamlit run app/streamlit_app.py")
    logger.info("  3. Open http://localhost:8501 in your browser")
    
    return best_model, results


if __name__ == "__main__":
    main()
