"""
Data Preprocessing Module

Handles cleaning, transformation, and feature engineering.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles all data preprocessing steps."""
    
    def __init__(
        self,
        numerical_features: list[str],
        categorical_features: list[str],
        target: str = "Churn"
    ):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.target = target
        self.preprocessor = None
        self.feature_names = None
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Convert TotalCharges to numeric (may contain empty strings)
        df['TotalCharges'] = pd.to_numeric(
            df['TotalCharges'], 
            errors='coerce'
        )
        
        # Fill missing TotalCharges with median
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
        
        # Drop customerID (not useful for modeling)
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])
        
        # Convert Yes/No columns to binary where appropriate
        yes_no_columns = [
            'Partner', 'Dependents', 'PhoneService', 
            'PaperlessBilling'
        ]
        for col in yes_no_columns:
            if col in df.columns:
                df[col] = df[col].map({'Yes': 1, 'No': 0})
        
        # Map target variable
        df[self.target] = df[self.target].map({'Yes': 1, 'No': 0})
        
        logger.info(f"Cleaned data shape: {df.shape}")
        return df
    
    def create_preprocessor(self) -> ColumnTransformer:
        """
        Create sklearn ColumnTransformer for preprocessing.
        
        Returns:
            Fitted ColumnTransformer
        """
        # Numerical pipeline
        numerical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, self.numerical_features),
                ('cat', categorical_pipeline, self.categorical_features)
            ]
        )
        
        return preprocessor
    
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit preprocessor on training data.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Self
        """
        self.preprocessor = self.create_preprocessor()
        self.preprocessor.fit(df)
        
        # Get feature names after transformation
        self._get_feature_names()
        
        logger.info("Preprocessor fitted")
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed numpy array
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        return self.preprocessor.transform(df)
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame to fit and transform
            
        Returns:
            Transformed numpy array
        """
        self.fit(df)
        return self.transform(df)
    
    def _get_feature_names(self) -> None:
        """Extract feature names after preprocessing."""
        if self.preprocessor is None:
            return
            
        # Get numerical feature names
        num_features = self.numerical_features
        
        # Get categorical feature names after one-hot encoding
        cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['encoder']
        cat_features = cat_encoder.get_feature_names_out(self.categorical_features).tolist()
        
        # Combine all feature names
        self.feature_names = num_features + cat_features
        logger.info(f"Total features after preprocessing: {len(self.feature_names)}")
    
    def save(self, filepath: str) -> None:
        """Save preprocessor to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'DataPreprocessor':
        """Load preprocessor from disk."""
        return joblib.load(filepath)


def prepare_data(
    df: pd.DataFrame,
    numerical_features: list[str],
    categorical_features: list[str],
    target: str = "Churn",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, DataPreprocessor]:
    """
    Full data preparation pipeline.
    
    Args:
        df: Raw input DataFrame
        numerical_features: List of numerical column names
        categorical_features: List of categorical column names
        target: Target column name
        test_size: Proportion for test set
        val_size: Proportion for validation set (from training)
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, val_df, test_df, preprocessor)
    """
    logger.info(f"Starting data preparation with {len(df)} samples")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        target=target
    )
    
    # Clean data
    df_clean = preprocessor.clean_data(df)
    
    # Separate features and target
    X = df_clean.drop(columns=[target])
    y = df_clean[target]
    
    # Split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Split: train vs val
    val_proportion = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_proportion, 
        random_state=random_state, stratify=y_trainval
    )
    
    logger.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Fit preprocessor on training data only
    preprocessor.fit(X_train)
    
    # Create processed DataFrames with feature names
    X_train_processed = pd.DataFrame(
        preprocessor.transform(X_train),
        columns=preprocessor.feature_names
    )
    X_train_processed[target] = y_train.values
    
    X_val_processed = pd.DataFrame(
        preprocessor.transform(X_val),
        columns=preprocessor.feature_names
    )
    X_val_processed[target] = y_val.values
    
    X_test_processed = pd.DataFrame(
        preprocessor.transform(X_test),
        columns=preprocessor.feature_names
    )
    X_test_processed[target] = y_test.values
    
    logger.info(f"Data preparation complete. Features: {len(preprocessor.feature_names)}")
    
    return X_train_processed, X_val_processed, X_test_processed, preprocessor


if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'src')
    import yaml
    
    # Load config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Load raw data
    df = pd.read_csv("data/raw/telco_churn.csv")
    
    # Prepare data
    train_df, val_df, test_df, preprocessor = prepare_data(
        df=df,
        numerical_features=config['features']['numerical'],
        categorical_features=config['features']['categorical'],
        target=config['features']['target'],
        test_size=config['training']['test_size'],
        val_size=config['training']['val_size'],
        random_state=config['training']['random_state']
    )
    
    # Save processed data
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    preprocessor.save("data/processed/preprocessor.pkl")
    
    print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Features: {len(preprocessor.feature_names)}")
