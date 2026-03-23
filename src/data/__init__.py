"""Data module for MLOps pipeline."""

from .ingestion import DataIngestion, ingest_data

__all__ = ["DataIngestion", "ingest_data"]
