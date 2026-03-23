"""
Full MLOps Pipeline

Orchestrates the complete ML pipeline from data ingestion to deployment.
"""

import sys
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLOpsPipeline:
    """Orchestrates the complete MLOps pipeline."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize pipeline with configuration."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.results = {
            'start_time': datetime.now().isoformat(),
            'stages': {},
            'artifacts': {}
        }
        
        logger.info("Pipeline initialized")
    
    def run_stage(self, stage_name: str, stage_func, *args, **kwargs) -> Any:
        """Run a pipeline stage with error handling."""
        logger.info(f"{'='*50}")
        logger.info(f"Starting stage: {stage_name}")
        logger.info(f"{'='*50}")
        
        start_time = datetime.now()
        
        try:
            result = stage_func(*args, **kwargs)
            
            self.results['stages'][stage_name] = {
                'status': 'success',
                'duration': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Stage {stage_name} completed successfully")
            return result
            
        except Exception as e:
            self.results['stages'][stage_name] = {
                'status': 'failed',
                'error': str(e),
                'duration': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.error(f"Stage {stage_name} failed: {e}")
            raise
    
    def stage_ingestion(self) -> Dict:
        """Stage 1: Data Ingestion."""
        from src.data.ingestion import ingest_data
        
        df = ingest_data(self.config)
        
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'path': str(Path(self.config['data']['raw_path']) / self.config['data']['churn_dataset']['filename'])
        }
    
    def stage_preprocessing(self) -> Dict:
        """Stage 2: Data Preprocessing."""
        import pandas as pd
        from src.data.preprocessing import prepare_data
        
        df = pd.read_csv(
            Path(self.config['data']['raw_path']) / 
            self.config['data']['churn_dataset']['filename']
        )
        
        train_df, val_df, test_df, preprocessor = prepare_data(
            df=df,
            numerical_features=self.config['features']['numerical'],
            categorical_features=self.config['features']['categorical'],
            target=self.config['features']['target'],
            test_size=self.config['training']['test_size'],
            val_size=self.config['training']['val_size'],
            random_state=self.config['training']['random_state']
        )
        
        # Save processed data
        Path(self.config['data']['processed_path']).mkdir(parents=True, exist_ok=True)
        train_df.to_csv(Path(self.config['data']['processed_path']) / "train.csv", index=False)
        val_df.to_csv(Path(self.config['data']['processed_path']) / "val.csv", index=False)
        test_df.to_csv(Path(self.config['data']['processed_path']) / "test.csv", index=False)
        preprocessor.save(str(Path(self.config['data']['processed_path']) / "preprocessor.pkl"))
        
        return {
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'features': len(preprocessor.feature_names)
        }
    
    def stage_training(self) -> Dict:
        """Stage 3: Model Training."""
        import pandas as pd
        from src.models.train import train_models
        
        train_df = pd.read_csv(Path(self.config['data']['processed_path']) / "train.csv")
        val_df = pd.read_csv(Path(self.config['data']['processed_path']) / "val.csv")
        
        best_model, best_metrics = train_models(
            train_df=train_df,
            val_df=val_df,
            config=self.config
        )
        
        return {
            'metrics': best_metrics,
            'model_path': "models/model.pkl"
        }
    
    def stage_evaluation(self) -> Dict:
        """Stage 4: Model Evaluation."""
        import pandas as pd
        from src.evaluation.evaluate import evaluate_model
        
        test_df = pd.read_csv(Path(self.config['data']['processed_path']) / "test.csv")
        
        report = evaluate_model(
            test_df=test_df,
            model_path="models/model.pkl",
            thresholds={
                'accuracy': 0.75,
                'roc_auc': 0.80,
                'recall': 0.50
            }
        )
        
        return {
            'deployment_ready': report.get('deployment_ready', False),
            'metrics': report['summary']
        }
    
    def stage_registry(self) -> Dict:
        """Stage 5: Model Registry."""
        from src.models.registry import register_best_model
        
        model_info = register_best_model(
            model_name=self.config['model']['name'],
            experiment_name=self.config['project']['name']
        )
        
        return model_info
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete MLOps pipeline."""
        logger.info("Starting full MLOps pipeline")
        
        # Stage 1: Data Ingestion
        ingestion_result = self.run_stage("ingestion", self.stage_ingestion)
        self.results['artifacts']['ingestion'] = ingestion_result
        
        # Stage 2: Preprocessing
        preprocessing_result = self.run_stage("preprocessing", self.stage_preprocessing)
        self.results['artifacts']['preprocessing'] = preprocessing_result
        
        # Stage 3: Training
        training_result = self.run_stage("training", self.stage_training)
        self.results['artifacts']['training'] = training_result
        
        # Stage 4: Evaluation
        evaluation_result = self.run_stage("evaluation", self.stage_evaluation)
        self.results['artifacts']['evaluation'] = evaluation_result
        
        # Stage 5: Registry (only if evaluation passed)
        if evaluation_result.get('deployment_ready', False):
            registry_result = self.run_stage("registry", self.stage_registry)
            self.results['artifacts']['registry'] = registry_result
        else:
            logger.warning("Model did not meet deployment thresholds. Skipping registry.")
            self.results['stages']['registry'] = {'status': 'skipped', 'reason': 'deployment thresholds not met'}
        
        # Finalize results
        self.results['end_time'] = datetime.now().isoformat()
        self.results['success'] = all(
            s.get('status') in ['success', 'skipped'] 
            for s in self.results['stages'].values()
        )
        
        # Save pipeline results
        self._save_results()
        
        return self.results
    
    def _save_results(self) -> None:
        """Save pipeline results."""
        results_path = Path("reports/pipeline_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert any non-serializable types
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif hasattr(obj, 'item'):  # numpy types
                return obj.item()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            return obj
        
        with open(results_path, 'w') as f:
            json.dump(convert(self.results), f, indent=2)
        
        logger.info(f"Pipeline results saved to {results_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MLOps Pipeline")
    parser.add_argument(
        "--stage", 
        choices=["ingestion", "preprocessing", "training", "evaluation", "registry", "full"],
        default="full",
        help="Pipeline stage to run"
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    pipeline = MLOpsPipeline(config_path=args.config)
    
    if args.stage == "full":
        results = pipeline.run_full_pipeline()
    else:
        stage_func = getattr(pipeline, f"stage_{args.stage}")
        results = pipeline.run_stage(args.stage, stage_func)
    
    # Print summary
    print("\n" + "="*50)
    print("PIPELINE SUMMARY")
    print("="*50)
    
    for stage, info in results.get('stages', {}).items():
        status = info.get('status', 'unknown')
        duration = info.get('duration', 0)
        symbol = "✓" if status == "success" else ("⊘" if status == "skipped" else "✗")
        print(f"{symbol} {stage}: {status} ({duration:.2f}s)")
    
    if 'artifacts' in results:
        print("\nArtifacts:")
        for name, artifact in results['artifacts'].items():
            print(f"  - {name}: {artifact}")
    
    print("="*50)
    
    return 0 if results.get('success', False) else 1


if __name__ == "__main__":
    sys.exit(main())
