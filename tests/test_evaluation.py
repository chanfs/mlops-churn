"""Tests for model evaluation module."""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import sys
sys.path.insert(0, 'src')

from src.evaluation.evaluate import ModelEvaluator


class TestModelEvaluator:
    """Test ModelEvaluator class."""
    
    @pytest.fixture
    def sample_model(self):
        """Create a simple trained model for testing."""
        X = np.random.randn(100, 3)
        y = np.random.randint(0, 2, 100)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        return model
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples),
            'Churn': np.random.randint(0, 2, n_samples)
        }
        return pd.DataFrame(data)
    
    def test_evaluate_returns_metrics(self, sample_model, sample_data):
        """Test that evaluation returns expected metrics."""
        evaluator = ModelEvaluator(sample_model)
        metrics = evaluator.evaluate(sample_data)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        
        # Check metrics are in valid range
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_evaluate_stores_confusion_matrix(self, sample_model, sample_data):
        """Test that confusion matrix is stored."""
        evaluator = ModelEvaluator(sample_model)
        evaluator.evaluate(sample_data)
        
        assert hasattr(evaluator, 'cm')
        assert evaluator.cm.shape == (2, 2)
    
    def test_evaluate_stores_roc_data(self, sample_model, sample_data):
        """Test that ROC curve data is stored."""
        evaluator = ModelEvaluator(sample_model)
        evaluator.evaluate(sample_data)
        
        assert hasattr(evaluator, 'fpr')
        assert hasattr(evaluator, 'tpr')
        assert len(evaluator.fpr) == len(evaluator.tpr)
    
    def test_check_thresholds(self, sample_model, sample_data):
        """Test threshold checking."""
        evaluator = ModelEvaluator(sample_model)
        evaluator.evaluate(sample_data)
        
        thresholds = {
            'accuracy': 0.3,  # Low threshold for test
            'roc_auc': 0.4
        }
        
        results = evaluator.check_thresholds(thresholds)
        
        assert 'accuracy' in results
        assert 'roc_auc' in results
        assert all(isinstance(v, bool) for v in results.values())
    
    def test_generate_report(self, sample_model, sample_data, tmp_path):
        """Test report generation."""
        # Change to temp directory for report writing
        import os
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            evaluator = ModelEvaluator(sample_model)
            evaluator.evaluate(sample_data)
            report = evaluator.generate_report()
            
            assert 'summary' in report
            assert 'classification_report' in report
            assert 'confusion_matrix' in report
            
            # Check report file was created
            report_path = tmp_path / "reports" / "evaluation_report.json"
            assert report_path.exists()
        finally:
            os.chdir(original_dir)
