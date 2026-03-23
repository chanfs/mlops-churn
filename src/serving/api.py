"""
Model Serving Module

FastAPI-based inference service.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="Customer Churn Prediction Service",
    version="0.1.0"
)


# Request/Response models
class ChurnPrediction(BaseModel):
    customer_id: Optional[str] = None
    churn_probability: float
    prediction: str
    confidence: float


class BatchPredictionRequest(BaseModel):
    records: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    model_version: Optional[str] = None
    timestamp: datetime


class ModelPredictor:
    """Handles model loading and predictions."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = None
            cls._instance.preprocessor = None
            cls._instance.feature_names = None
        return cls._instance
    
    def load_model(self, model_path: str = "models/model.pkl") -> None:
        """Load model and preprocessor."""
        logger.info(f"Loading model from {model_path}")
        
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = joblib.load(model_file)
        
        # Try to load preprocessor
        preprocessor_path = "data/processed/preprocessor.pkl"
        if Path(preprocessor_path).exists():
            self.preprocessor = joblib.load(preprocessor_path)
            self.feature_names = self.preprocessor.feature_names
        
        logger.info("Model loaded successfully")
    
    def predict(self, features: Dict[str, Any]) -> ChurnPrediction:
        """Make single prediction."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Preprocess if needed
        if self.preprocessor is not None:
            X = self.preprocessor.transform(df)
            feature_cols = self.preprocessor.feature_names
        else:
            X = df
            feature_cols = list(df.columns)
        
        # Predict
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0, 1]
        
        return ChurnPrediction(
            prediction="Churn" if prediction == 1 else "No Churn",
            churn_probability=float(probability),
            confidence=float(max(probability, 1 - probability))
        )
    
    def predict_batch(self, records: List[Dict[str, Any]]) -> List[ChurnPrediction]:
        """Make batch predictions."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Preprocess
        if self.preprocessor is not None:
            X = self.preprocessor.transform(df)
        else:
            X = df
        
        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append(ChurnPrediction(
                prediction="Churn" if pred == 1 else "No Churn",
                churn_probability=float(prob),
                confidence=float(max(prob, 1 - prob))
            ))
        
        return results


# Global predictor
predictor = ModelPredictor()


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    predictor.load_model()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_version="1.0.0",
        timestamp=datetime.now()
    )


@app.post("/predict", response_model=ChurnPrediction)
async def predict(features: Dict[str, Any]):
    """
    Make a single churn prediction.
    
    Args:
        features: Customer features as JSON
        
    Returns:
        Churn prediction with probability
    """
    try:
        result = predictor.predict(features)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=List[ChurnPrediction])
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions.
    
    Args:
        request: List of customer records
        
    Returns:
        List of predictions
    """
    try:
        results = predictor.predict_batch(request.records)
        return results
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    """Get model information."""
    return {
        "name": "churn_classifier",
        "version": "1.0.0",
        "type": "RandomForestClassifier",
        "features": predictor.feature_names
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
