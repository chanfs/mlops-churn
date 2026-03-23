"""Serving module for MLOps pipeline."""

from .api import app, ModelPredictor

__all__ = ["app", "ModelPredictor"]
