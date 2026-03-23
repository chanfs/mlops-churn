"""
Model Registry Module

Handles model registration, versioning, and stage transitions.
"""

import mlflow
from mlflow.entities.model_registry import ModelVersion
from typing import Optional, List
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """MLflow Model Registry wrapper."""
    
    # Model stage names
    STAGE_NONE = "None"
    STAGE_STAGING = "Staging"
    STAGE_PRODUCTION = "Production"
    STAGE_ARCHIVED = "Archived"
    
    def __init__(
        self,
        experiment_name: str = "churn_prediction",
        tracking_uri: Optional[str] = None
    ):
        """
        Initialize model registry.
        
        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking URI
        """
        self.experiment_name = experiment_name
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri("mlruns")
        
        # Set registry URI
        mlflow.set_registry_uri(mlflow.get_tracking_uri())
        
        logger.info(f"Model Registry initialized")
        logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
    
    def register_model(
        self,
        model_uri: str,
        model_name: str,
        run_id: Optional[str] = None,
        tags: Optional[dict] = None
    ) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            model_uri: URI of the model to register (e.g., "runs:/run_id/model")
            model_name: Name for the registered model
            run_id: Optional MLflow run ID
            tags: Optional tags to add to the model
            
        Returns:
            ModelVersion object
        """
        logger.info(f"Registering model: {model_name}")
        logger.info(f"Model URI: {model_uri}")
        
        # Register model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        # Add tags if provided
        if tags:
            client = mlflow.tracking.MlflowClient()
            for key, value in tags.items():
                client.set_registered_model_tag(model_name, key, value)
        
        logger.info(f"Model registered: {model_name} version {model_version.version}")
        
        return model_version
    
    def transition_stage(
        self,
        model_name: str,
        version: str,
        new_stage: str
    ) -> ModelVersion:
        """
        Transition model to a new stage.
        
        Args:
            model_name: Registered model name
            version: Model version number
            new_stage: Target stage (Staging/Production/Archived)
            
        Returns:
            Updated ModelVersion object
        """
        client = mlflow.tracking.MlflowClient()
        
        logger.info(f"Transitioning {model_name} v{version} to {new_stage}")
        
        # Archive existing models in target stage
        if new_stage == self.STAGE_PRODUCTION:
            self._archive_production_models(model_name)
        
        # Transition model
        model_version = client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=new_stage
        )
        
        logger.info(f"Model {model_name} v{version} now in {new_stage}")
        
        return model_version
    
    def _archive_production_models(self, model_name: str) -> None:
        """Archive all models currently in Production stage."""
        client = mlflow.tracking.MlflowClient()
        
        # Get all versions
        versions = client.search_model_versions(
            f"name='{model_name}' AND stage='{self.STAGE_PRODUCTION}'"
        )
        
        for version in versions:
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage=self.STAGE_ARCHIVED
            )
            logger.info(f"Archived {model_name} v{version.version}")
    
    def get_latest_version(
        self,
        model_name: str,
        stage: Optional[str] = None
    ) -> Optional[ModelVersion]:
        """
        Get the latest model version.
        
        Args:
            model_name: Registered model name
            stage: Optional stage filter
            
        Returns:
            Latest ModelVersion or None
        """
        client = mlflow.tracking.MlflowClient()
        
        filter_query = f"name='{model_name}'"
        if stage:
            filter_query += f" AND stage='{stage}'"
        
        versions = client.search_model_versions(
            filter_query,
            order_by=["version_number DESC"]
        )
        
        if not versions:
            return None
        
        return versions[0]
    
    def get_production_model(self, model_name: str) -> Optional[ModelVersion]:
        """Get the current production model version."""
        return self.get_latest_version(model_name, self.STAGE_PRODUCTION)
    
    def get_staging_model(self, model_name: str) -> Optional[ModelVersion]:
        """Get the current staging model version."""
        return self.get_latest_version(model_name, self.STAGE_STAGING)
    
    def list_models(self) -> List[dict]:
        """
        List all registered models with their versions.
        
        Returns:
            List of model info dictionaries
        """
        client = mlflow.tracking.MlflowClient()
        models = client.search_registered_models()
        
        result = []
        for model in models:
            versions = client.search_model_versions(f"name='{model.name}'")
            
            model_info = {
                'name': model.name,
                'tags': dict(model.tags),
                'versions': [
                    {
                        'version': v.version,
                        'stage': v.current_stage,
                        'run_id': v.run_id,
                        'created': v.creation_timestamp
                    }
                    for v in versions
                ]
            }
            result.append(model_info)
        
        return result
    
    def delete_model(self, model_name: str) -> None:
        """Delete a registered model and all versions."""
        client = mlflow.tracking.MlflowClient()
        logger.info(f"Deleting model: {model_name}")
        client.delete_registered_model(model_name)
    
    def promote_to_production(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> ModelVersion:
        """
        Promote a model version to Production.
        
        Args:
            model_name: Registered model name
            version: Version to promote (latest if not specified)
            
        Returns:
            Promoted ModelVersion
        """
        if version is None:
            # Get latest staging model
            staging_model = self.get_staging_model(model_name)
            if staging_model is None:
                # No staging model, get latest overall
                latest = self.get_latest_version(model_name)
                if latest is None:
                    raise ValueError(f"No models found for {model_name}")
                version = latest.version
            else:
                version = staging_model.version
        
        return self.transition_stage(
            model_name=model_name,
            version=str(version),
            new_stage=self.STAGE_PRODUCTION
        )


def register_best_model(
    model_name: str = "churn_classifier",
    experiment_name: str = "churn_prediction",
    metric: str = "roc_auc"
) -> dict:
    """
    Register the best model from an experiment.
    
    Args:
        model_name: Name for the registered model
        experiment_name: MLflow experiment name
        metric: Metric to optimize
        
    Returns:
        Model info dictionary
    """
    logger.info(f"Finding best model by {metric}")
    
    # Initialize registry
    registry = ModelRegistry(experiment_name=experiment_name)
    
    # Get best run
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=[f"metrics.{metric} DESC"]
    )
    
    if len(runs) == 0:
        raise ValueError(f"No runs found in experiment {experiment_name}")
    
    best_run = runs.iloc[0]
    run_id = best_run['run_id']
    best_metric = best_run[f'metrics.{metric}']
    
    logger.info(f"Best run: {run_id} with {metric}={best_metric:.4f}")
    
    # Register model
    model_uri = f"runs:/{run_id}/model"
    
    model_version = registry.register_model(
        model_uri=model_uri,
        model_name=model_name,
        run_id=run_id,
        tags={
            "best_metric": metric,
            "metric_value": str(best_metric),
            "run_id": run_id
        }
    )
    
    # Transition to staging
    registry.transition_stage(
        model_name=model_name,
        version=model_version.version,
        new_stage=registry.STAGE_STAGING
    )
    
    logger.info(f"Model registered and moved to Staging")
    
    return {
        'model_name': model_name,
        'version': model_version.version,
        'run_id': run_id,
        'metric': metric,
        'value': best_metric,
        'stage': registry.STAGE_STAGING
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'src')
    
    # Register best model
    result = register_best_model()
    
    print("\n=== Model Registration ===")
    print(f"Model Name: {result['model_name']}")
    print(f"Version: {result['version']}")
    print(f"Run ID: {result['run_id']}")
    print(f"Best {result['metric']}: {result['value']:.4f}")
    print(f"Stage: {result['stage']}")
    
    # List all models
    registry = ModelRegistry()
    models = registry.list_models()
    
    print("\n=== Registered Models ===")
    for model in models:
        print(f"\n{model['name']}:")
        for v in model['versions']:
            print(f"  v{v['version']} - {v['stage']} (run: {v['run_id'][:8]}...)")
