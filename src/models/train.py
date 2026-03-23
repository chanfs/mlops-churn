"""
Model Training Module

Handles model training with MLflow experiment tracking.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Optional, Any
import logging

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training and MLflow logging."""
    
    def __init__(
        self,
        experiment_name: str = "churn_prediction",
        tracking_uri: Optional[str] = None
    ):
        self.experiment_name = experiment_name
        
        # Set tracking URI (local file or remote server)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri("mlruns")
        
        # Set or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"MLflow experiment: {experiment_name}")
        logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
    
    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict[str, Any],
        run_name: str = "random_forest"
    ) -> tuple[RandomForestClassifier, dict]:
        """
        Train Random Forest model with MLflow tracking.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            params: Model hyperparameters
            run_name: Name for MLflow run
            
        Returns:
            Tuple of (trained model, metrics dict)
        """
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_params(params)
            
            # Train model
            rf_params = params.copy()
            rf_params.pop('random_state', None)  # Remove to avoid duplicate
            model = RandomForestClassifier(**rf_params, random_state=42)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred),
                'recall': recall_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred),
                'roc_auc': roc_auc_score(y_val, y_pred_proba)
            }
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=X_train.head(1)
            )
            
            # Log feature importance
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Save as artifact
                importance_path = Path("feature_importance.csv")
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(str(importance_path))
                importance_path.unlink()
            
            logger.info(f"Random Forest trained - Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
            
            return model, metrics
    
    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict[str, Any],
        run_name: str = "logistic_regression"
    ) -> tuple[LogisticRegression, dict]:
        """
        Train Logistic Regression model with MLflow tracking.
        """
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_params(params)
            
            # Train model
            model = LogisticRegression(**params, random_state=42, max_iter=1000)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred),
                'recall': recall_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred),
                'roc_auc': roc_auc_score(y_val, y_pred_proba)
            }
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=X_train.head(1)
            )
            
            logger.info(f"Logistic Regression trained - Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
            
            return model, metrics
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict[str, Any],
        run_name: str = "xgboost"
    ) -> tuple[Any, dict]:
        """
        Train XGBoost model with MLflow tracking.
        """
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("XGBoost not installed. Skipping XGBoost training.")
            return None, {}
        
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_params(params)
            
            # Create datasets
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            # Train model
            model = xgb.train(
                params,
                dtrain,
                evals=[(dtrain, 'train'), (dval, 'val')],
                num_boost_round=params.get('n_estimators', 100),
                verbose_eval=False
            )
            
            # Predictions
            y_pred = (model.predict(dval) > 0.5).astype(int)
            y_pred_proba = model.predict(dval)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred),
                'recall': recall_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred),
                'roc_auc': roc_auc_score(y_val, y_pred_proba)
            }
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.xgboost.log_model(model, "model")
            
            logger.info(f"XGBoost trained - Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
            
            return model, metrics
    
    def get_best_run(self, metric: str = "roc_auc") -> Optional[mlflow.entities.Run]:
        """
        Get the best run based on a metric.
        
        Args:
            metric: Metric to optimize
            
        Returns:
            Best MLflow run
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} DESC"]
        )
        
        if len(runs) == 0:
            return None
        
        best_run_id = runs.iloc[0]['run_id']
        return mlflow.get_run(best_run_id)
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all trained models.
        
        Returns:
            DataFrame with model comparison
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=["metrics.roc_auc DESC"]
        )
        
        if len(runs) == 0:
            return pd.DataFrame()
        
        # Select relevant columns
        columns = ['run_id', 'tags.mlflow.runName', 
                   'metrics.accuracy', 'metrics.precision', 
                   'metrics.recall', 'metrics.f1', 'metrics.roc_auc']
        
        return runs[columns].rename(columns={'tags.mlflow.runName': 'model'})


def train_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict,
    tracking_uri: Optional[str] = None
) -> tuple[Any, dict]:
    """
    Train multiple models and return the best one.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        config: Configuration dictionary
        tracking_uri: MLflow tracking URI
        
    Returns:
        Tuple of (best model, best metrics)
    """
    logger.info("Starting model training")
    
    # Separate features and target
    target = config['features']['target']
    feature_cols = [c for c in train_df.columns if c != target]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target]
    X_val = val_df[feature_cols]
    y_val = val_df[target]
    
    # Initialize trainer
    trainer = ModelTrainer(
        experiment_name=config['project']['name'],
        tracking_uri=tracking_uri
    )
    
    all_models = {}
    
    # Train Random Forest
    logger.info("Training Random Forest...")
    rf_params = config['model']['params']
    rf_model, rf_metrics = trainer.train_random_forest(
        X_train, y_train, X_val, y_val, rf_params
    )
    all_models['random_forest'] = (rf_model, rf_metrics)
    
    # Train Logistic Regression
    logger.info("Training Logistic Regression...")
    lr_params = {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'lbfgs'
    }
    lr_model, lr_metrics = trainer.train_logistic_regression(
        X_train, y_train, X_val, y_val, lr_params
    )
    all_models['logistic_regression'] = (lr_model, lr_metrics)
    
    # Compare models
    comparison_df = trainer.compare_models()
    logger.info("\nModel Comparison:")
    logger.info(comparison_df[['model', 'metrics.roc_auc']].to_string())
    
    # Get best model
    best_model_name = max(all_models.keys(), 
                          key=lambda k: all_models[k][1].get('roc_auc', 0))
    best_model, best_metrics = all_models[best_model_name]
    
    logger.info(f"\nBest model: {best_model_name}")
    logger.info(f"Best ROC-AUC: {best_metrics['roc_auc']:.4f}")
    
    # Save best model
    model_path = Path("models/model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    logger.info(f"Best model saved to {model_path}")
    
    # Save metrics
    metrics_path = Path("metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(best_metrics, f, indent=2)
    
    return best_model, best_metrics


if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'src')
    import yaml
    
    # Load config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Load processed data
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    
    # Train models
    best_model, best_metrics = train_models(train_df, val_df, config)
    
    print(f"\nBest Metrics:")
    for metric, value in best_metrics.items():
        print(f"  {metric}: {value:.4f}")
