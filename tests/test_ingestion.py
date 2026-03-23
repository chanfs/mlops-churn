"""Tests for data ingestion module."""

import pytest
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, 'src')

from src.data.ingestion import DataIngestion, ChurnDataSchema


class TestDataIngestion:
    """Test DataIngestion class."""
    
    def test_init_creates_directory(self, tmp_path):
        """Test that initialization creates the raw data directory."""
        ingestion = DataIngestion(str(tmp_path / "data"))
        assert ingestion.raw_data_path.exists()
    
    def test_validate_schema_valid_data(self):
        """Test schema validation with valid data."""
        ingestion = DataIngestion()
        
        # Create minimal valid DataFrame
        df = pd.DataFrame({
            'customerID': ['001'],
            'gender': ['Female'],
            'SeniorCitizen': [0],
            'Partner': ['Yes'],
            'Dependents': ['No'],
            'tenure': [12],
            'PhoneService': ['Yes'],
            'MultipleLines': ['No phone service'],
            'InternetService': ['DSL'],
            'OnlineSecurity': ['No'],
            'OnlineBackup': ['Yes'],
            'DeviceProtection': ['No'],
            'TechSupport': ['No'],
            'StreamingTV': ['Yes'],
            'StreamingMovies': ['Yes'],
            'Contract': ['Month-to-month'],
            'PaperlessBilling': ['Yes'],
            'PaymentMethod': ['Electronic check'],
            'MonthlyCharges': [29.85],
            'TotalCharges': ['29.85'],
            'Churn': ['No']
        })
        
        is_valid, errors = ingestion.validate_schema(df)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_schema_missing_columns(self):
        """Test schema validation with missing columns."""
        ingestion = DataIngestion()
        
        df = pd.DataFrame({
            'customerID': ['001'],
            'Churn': ['No']
        })
        
        is_valid, errors = ingestion.validate_schema(df)
        assert not is_valid
        assert any('Missing columns' in e for e in errors)
    
    def test_validate_schema_invalid_churn_values(self):
        """Test schema validation with invalid Churn values."""
        ingestion = DataIngestion()
        
        df = pd.DataFrame({
            'customerID': ['001'],
            'Churn': ['Maybe'],
            'gender': ['Female'],
            'SeniorCitizen': [0],
            'Partner': ['Yes'],
            'Dependents': ['No'],
            'tenure': [12],
            'PhoneService': ['Yes'],
            'MultipleLines': ['No'],
            'InternetService': ['DSL'],
            'OnlineSecurity': ['No'],
            'OnlineBackup': ['Yes'],
            'DeviceProtection': ['No'],
            'TechSupport': ['No'],
            'StreamingTV': ['Yes'],
            'StreamingMovies': ['Yes'],
            'Contract': ['Month-to-month'],
            'PaperlessBilling': ['Yes'],
            'PaymentMethod': ['Electronic check'],
            'MonthlyCharges': [29.85],
            'TotalCharges': ['29.85']
        })
        
        is_valid, errors = ingestion.validate_schema(df)
        assert not is_valid
        assert any('Invalid Churn' in e for e in errors)
    
    def test_get_data_info(self):
        """Test data info generation."""
        ingestion = DataIngestion()
        
        df = pd.DataFrame({
            'customerID': ['001', '002'],
            'Churn': ['No', 'Yes'],
            'tenure': [12, 24]
        })
        
        info = ingestion.get_data_info(df)
        
        assert info['num_rows'] == 2
        assert info['num_columns'] == 3
        assert info['churn_distribution'] == {'No': 1, 'Yes': 1}
