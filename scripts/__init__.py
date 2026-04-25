"""
Final System - Agentic Fraud Detection
=======================================

A clean, simple MAPE-K implementation without LangGraph/LangChain.

Usage:
    from pipeline import run_pipeline
    result = run_pipeline("data.csv", target_col="is_fraud")
"""

from .pipeline import MAPEKPipeline, run_pipeline
from .agents.base import BaseAgent, AgentResult
from .agents.drift_agent import DriftAgent
from .agents.balance_agent import BalanceAgent
from .agents.training_agent import TrainingAgent, fgsm_attack
from .agents.evaluation_agent import EvaluationAgent

__version__ = "1.0.0"
__all__ = [
    "MAPEKPipeline",
    "run_pipeline",
    "BaseAgent",
    "AgentResult",
    "DriftAgent",
    "BalanceAgent",
    "TrainingAgent",
    "EvaluationAgent",
    "fgsm_attack"
]