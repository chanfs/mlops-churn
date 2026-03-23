"""
Feature Engineering Module

Creates additional features from raw data.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create additional features for the model."""
    
    def __init__(self):
        self.feature_metadata = {}
    
    def create_tenure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on tenure."""
        df = df.copy()
        
        # Tenure groups
        df['tenure_group'] = pd.cut(
            df['tenure'],
            bins=[0, 12, 24, 48, 72, float('inf')],
            labels=['0-12', '13-24', '25-48', '49-72', '72+']
        )
        
        # Tenure squared (for non-linear relationships)
        df['tenure_squared'] = df['tenure'] ** 2
        
        # Is new customer (tenure < 6 months)
        df['is_new_customer'] = (df['tenure'] < 6).astype(int)
        
        # Is long-term customer (tenure > 48 months)
        df['is_long_term'] = (df['tenure'] > 48).astype(int)
        
        logger.info("Created tenure-based features")
        return df
    
    def create_charge_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on charges."""
        df = df.copy()
        
        # Average monthly charge estimate
        df['avg_monthly_charge'] = df['TotalCharges'] / (df['tenure'] + 1)
        
        # Charge per tenure ratio
        df['charge_tenure_ratio'] = df['MonthlyCharges'] / (df['tenure'] + 1)
        
        # High monthly charge flag (above median)
        median_charge = df['MonthlyCharges'].median()
        df['high_monthly_charge'] = (df['MonthlyCharges'] > median_charge).astype(int)
        
        logger.info("Created charge-based features")
        return df
    
    def create_service_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Count number of services subscribed."""
        df = df.copy()
        
        service_columns = [
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        # Count services (Yes = 1, No = 0, No phone/internet = 0)
        def count_services(row):
            count = 0
            for col in service_columns:
                if col in df.columns:
                    val = row[col]
                    if isinstance(val, (int, float)):
                        count += 1 if val > 0 else 0
                    elif isinstance(val, str):
                        if val not in ['No', 'No phone service', 'No internet service']:
                            count += 1
            return count
        
        df['service_count'] = df.apply(count_services, axis=1)
        
        # Has multiple services flag
        df['has_multiple_services'] = (df['service_count'] > 3).astype(int)
        
        logger.info("Created service count features")
        return df
    
    def create_contract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on contract type."""
        df = df.copy()
        
        # Is month-to-month (higher churn risk)
        df['is_month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)
        
        # Is long-term contract
        df['is_long_contract'] = df['Contract'].isin(['One year', 'Two year']).astype(int)
        
        logger.info("Created contract-based features")
        return df
    
    def create_payment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on payment method."""
        df = df.copy()
        
        # Is electronic payment (higher churn risk)
        df['is_electronic_payment'] = (
            df['PaymentMethod'] == 'Electronic check'
        ).astype(int)
        
        # Is automatic payment (lower churn risk)
        df['is_automatic_payment'] = df['PaymentMethod'].isin([
            'Bank transfer (automatic)',
            'Credit card (automatic)'
        ]).astype(int)
        
        logger.info("Created payment-based features")
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        df = self.create_tenure_features(df)
        df = self.create_charge_features(df)
        df = self.create_service_count(df)
        df = self.create_contract_features(df)
        df = self.create_payment_features(df)
        
        logger.info(f"Feature engineering complete. Total columns: {len(df.columns)}")
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering (same as fit_transform for stateless transforms)."""
        return self.fit_transform(df)


if __name__ == "__main__":
    # Test feature engineering
    df = pd.read_csv("data/raw/telco_churn.csv")
    
    engineer = FeatureEngineer()
    df_engineered = engineer.fit_transform(df)
    
    print(f"\nOriginal columns: {len(df.columns)}")
    print(f"Engineered columns: {len(df_engineered.columns)}")
    print(f"\nNew features added:")
    new_features = set(df_engineered.columns) - set(df.columns)
    for feat in sorted(new_features):
        print(f"  - {feat}")
