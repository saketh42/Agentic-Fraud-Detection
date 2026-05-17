"""Lazy imports to avoid slow startup on WSL/Windows filesystem."""

def __getattr__(name):
    if name == 'BaseAgent' or name == 'AgentResult':
        from .base import BaseAgent, AgentResult
        return BaseAgent if name == 'BaseAgent' else AgentResult
    elif name == 'MonitorAgent':
        from .monitor_agent import MonitorAgent
        return MonitorAgent
    elif name == 'FeatureExtractionAgent':
        from .feature_extraction_agent import FeatureExtractionAgent
        return FeatureExtractionAgent
    elif name == 'ContextRetrievalAgent':
        from .context_retrieval_agent import ContextRetrievalAgent
        return ContextRetrievalAgent
    elif name == 'FraudScoringAgent':
        from .fraud_scoring_agent import FraudScoringAgent
        return FraudScoringAgent
    elif name == 'LLMReasoningAgent':
        from .llm_reasoning_agent import LLMReasoningAgent
        return LLMReasoningAgent
    elif name == 'PatternLearningAgent':
        from .pattern_learning_agent import PatternLearningAgent
        return PatternLearningAgent
    elif name == 'AdversarialSimulationAgent':
        from .adversarial_simulation_agent import AdversarialSimulationAgent
        return AdversarialSimulationAgent
    elif name == 'PlanningAgent':
        from .planning_agent import PlanningAgent
        return PlanningAgent
    elif name == 'ExecuteAgent':
        from .execute_agent import ExecuteAgent
        return ExecuteAgent
    elif name == 'FeedbackAgent':
        from .feedback_agent import FeedbackAgent
        return FeedbackAgent
    elif name == 'KnowledgeStore':
        from .knowledge_store import KnowledgeStore
        return KnowledgeStore
    elif name == 'DriftAgent':
        from .drift_agent import DriftAgent
        return DriftAgent
    elif name == 'BalanceAgent':
        from .balance_agent import BalanceAgent
        return BalanceAgent
    elif name == 'TrainingAgent':
        from .training_agent import TrainingAgent
        return TrainingAgent
    elif name == 'fgsm_attack':
        from .training_agent import fgsm_attack
        return fgsm_attack
    elif name == 'EvaluationAgent':
        from .evaluation_agent import EvaluationAgent
        return EvaluationAgent
    elif name == 'IngestionAgent':
        from .ingestion_agent import IngestionAgent
        return IngestionAgent
    elif name == 'ScraperAgent':
        from .ingestion_agent import ScraperAgent
        return ScraperAgent
    elif name == 'AnnotationAgent':
        from .ingestion_agent import AnnotationAgent
        return AnnotationAgent
    elif name == 'DecisionAgent':
        from .decision_agent import DecisionAgent
        return DecisionAgent
    elif name == 'CriticAgent':
        from .critic_agent import CriticAgent
        return CriticAgent
    elif name == 'MetricsTracker':
        from .metrics_tracker import MetricsTracker
        return MetricsTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'BaseAgent', 'AgentResult',
    'MonitorAgent', 'FeatureExtractionAgent', 'ContextRetrievalAgent',
    'FraudScoringAgent', 'LLMReasoningAgent', 'PatternLearningAgent',
    'AdversarialSimulationAgent', 'PlanningAgent', 'ExecuteAgent',
    'FeedbackAgent', 'KnowledgeStore',
    'DriftAgent', 'BalanceAgent', 'TrainingAgent', 'EvaluationAgent',
    'IngestionAgent', 'ScraperAgent', 'AnnotationAgent',
    'DecisionAgent', 'CriticAgent', 'MetricsTracker',
    'fgsm_attack',
]
