"""
Final System - Agentic Fraud Detection
======================================= 

A clean, simple MAPE-K implementation without LangGraph/LangChain.

Usage:
    from scripts.pipeline import run_pipeline
    result = run_pipeline("data.csv", target_col="is_fraud")
"""

from .pipeline import MAPEKPipeline, run_pipeline

__version__ = "1.0.0"
__all__ = [
    "MAPEKPipeline",
    "run_pipeline",
]