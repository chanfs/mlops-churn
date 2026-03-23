"""
Data Ingestion Module

Handles downloading, loading, and validating raw data.
"""

import pandas as pd
import requests
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnDataSchema(BaseModel):
    """Schema for Telco Churn dataset validation."""
    
    customerID: str
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: str  # Can be numeric or empty string
    Churn: str = Field(pattern="^(Yes|No)$")


class DataIngestion:
    """Handles data download and initial validation."""
    
    def __init__(self, raw_data_path: str = "data/raw"):
        self.raw_data_path = Path(raw_data_path)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        
    def download_dataset(self, url: str, filename: str) -> Path:
        """
        Download dataset from URL to raw data directory.
        
        Args:
            url: Source URL for the dataset
            filename: Local filename to save
            
        Returns:
            Path to downloaded file
        """
        filepath = self.raw_data_path / filename
        
        if filepath.exists():
            logger.info(f"Dataset already exists at {filepath}")
            return filepath
            
        logger.info(f"Downloading dataset from {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
            
        logger.info(f"Dataset saved to {filepath}")
        return filepath
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load dataset from raw data directory.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            DataFrame with raw data
        """
        filepath = self.raw_data_path / filename
        logger.info(f"Loading data from {filepath}")
        
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def validate_schema(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """
        Validate DataFrame against expected schema.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Check required columns
        expected_columns = list(ChurnDataSchema.model_fields.keys())
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing columns: {missing_columns}")
            return False, errors
        
        # Check Churn values
        invalid_churn = set(df['Churn'].unique()) - {'Yes', 'No'}
        if invalid_churn:
            errors.append(f"Invalid Churn values: {invalid_churn}")
        
        # Check SeniorCitizen values
        if not df['SeniorCitizen'].isin([0, 1]).all():
            errors.append("SeniorCitizen must be 0 or 1")
        
        # Check for negative tenure
        if (df['tenure'] < 0).any():
            errors.append("Negative tenure values found")
        
        # Check MonthlyCharges
        if (df['MonthlyCharges'] < 0).any():
            errors.append("Negative MonthlyCharges values found")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get basic information about the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with dataset statistics
        """
        info = {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'churn_distribution': df['Churn'].value_counts().to_dict(),
        }
        return info


def ingest_data(config: dict) -> pd.DataFrame:
    """
    Main ingestion function using config.
    
    Args:
        config: Configuration dictionary with data settings
        
    Returns:
        Validated DataFrame
    """
    ingestion = DataIngestion()
    
    # Download
    filepath = ingestion.download_dataset(
        url=config['data']['churn_dataset']['url'],
        filename=config['data']['churn_dataset']['filename']
    )
    
    # Load
    df = ingestion.load_data(config['data']['churn_dataset']['filename'])
    
    # Validate
    is_valid, errors = ingestion.validate_schema(df)
    if not is_valid:
        raise ValueError(f"Data validation failed: {errors}")
    
    logger.info("Data validation passed")
    
    # Log info
    info = ingestion.get_data_info(df)
    logger.info(f"Dataset: {info['num_rows']} rows, {info['num_columns']} columns")
    logger.info(f"Churn distribution: {info['churn_distribution']}")
    
    return df


if __name__ == "__main__":
    import yaml
    
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    df = ingest_data(config)
    print(f"\nLoaded {len(df)} records")
    print(f"\nFirst few rows:\n{df.head()}")
