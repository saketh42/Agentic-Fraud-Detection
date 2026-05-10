"""
Pydantic models for the fraud detection API
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd


class TransactionFeatures(BaseModel):
    """Input features for a single transaction"""
    # Fraud labels
    transaction_upi_fraud: int = Field(0, description="UPI fraud indicator (0/1)")
    transaction_card_fraud: int = Field(0, description="Card fraud indicator (0/1)")
    transaction_bank_transfer: int = Field(0, description="Bank transfer fraud indicator (0/1)")
    commerce_nondelivery: int = Field(0, description="Non-delivery indicator (0/1)")
    commerce_fake_seller: int = Field(0, description="Fake seller indicator (0/1)")
    credential_phishing: int = Field(0, description="Phishing indicator (0/1)")
    social_authority_scam: int = Field(0, description="Authority scam indicator (0/1)")
    social_urgency_scam: int = Field(0, description="Urgency scam indicator (0/1)")
    meta_victim_story: int = Field(0, description="Victim story indicator (0/1)")
    meta_fraud_question: int = Field(0, description="Fraud question indicator (0/1)")
    
    # Key features
    payment_method: str = Field("unknown", description="Payment method")
    fraud_channel: str = Field("unknown", description="Channel used")
    victim_action: str = Field("unknown", description="Victim action")
    request_type: str = Field("unknown", description="Request type")
    impersonated_entity: str = Field("unknown", description="Impersonated entity")
    amount_mentioned: str = Field("unknown", description="Amount mentioned")
    currency: str = Field("unknown", description="Currency")
    urgency_level: str = Field("unknown", description="Urgency level")
    
    # Psychological tactics scores (0-1)
    urgency: float = Field(0.0, ge=0.0, le=1.0, description="Urgency score")
    fear: float = Field(0.0, ge=0.0, le=1.0, description="Fear score")
    authority: float = Field(0.0, ge=0.0, le=1.0, description="Authority score")
    reward: float = Field(0.0, ge=0.0, le=1.0, description="Reward score")
    
    # Amount features
    amount_normalized: float = Field(0.0, description="Normalized amount")
    has_amount: int = Field(0, description="Has amount flag (0/1)")


class PredictionResult(BaseModel):
    """Result of fraud prediction"""
    is_fraud: bool = Field(description="Fraud prediction (True/False)")
    fraud_probability: float = Field(description="Probability of fraud (0-1)")
    risk_level: str = Field(description="Risk level (HIGH/MEDIUM/LOW/MINIMAL)")
    model_confidence: float = Field(description="Model confidence in prediction")
    fraud_score: float = Field(description="Fraud score (0-1)")
    label_score: float = Field(description="Label-based score")
    tactic_score: float = Field(description="Psychological tactic score")
    feature_score: float = Field(description="Feature-based score")
    risk_features: List[Dict[str, Any]] = Field(description="Risk features breakdown")
    rule_breakdown: Dict[str, float] = Field(description="Rule-based scoring breakdown")
    pattern_name: str = Field(description="Detected fraud pattern")
    pattern_type: str = Field(description="Type of fraud pattern")
    pattern_confidence: float = Field(description="Confidence in pattern detection")


class BatchPredictionRequest(BaseModel):
    """Request for batch prediction"""
    transactions: List[TransactionFeatures] = Field(description="List of transactions")


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction"""
    predictions: List[PredictionResult] = Field(description="List of predictions")
    summary: Dict[str, Any] = Field(description="Summary statistics")


class SystemStatus(BaseModel):
    """System status information"""
    status: str = Field(description="System status (ok/error)")
    model_loaded: bool = Field(description="Is model loaded")
    last_trained: str = Field(description="Timestamp of last training")
    pipeline_ready: bool = Field(description="Is pipeline ready")
    drift_detected: bool = Field(description="Was drift detected in last run")
    metrics: Dict[str, float] = Field(description="Latest evaluation metrics")


class MetricsResponse(BaseModel):
    """Model metrics response"""
    accuracy: float = Field(description="Model accuracy")
    precision: float = Field(description="Model precision")
    recall: float = Field(description="Model recall")
    f1: float = Field(description="Model F1 score")
    roc_auc: float = Field(description="Model ROC-AUC score")
    true_negatives: int = Field(description="True negatives")
    false_positives: int = Field(description="False positives")
    false_negatives: int = Field(description="False negatives")
    true_positives: int = Field(description="True positives")
    fpr: float = Field(description="False positive rate")
    robustness_curve: List[Dict[str, float]] = Field(description="Robustness curve data")
    clean_f1: float = Field(description="Clean F1 score")
    worst_f1: float = Field(description="Worst F1 under attack")
    avg_f1: float = Field(description="Average F1 under attack")
    f1_drop: float = Field(description="F1 drop under attack")
    is_robust: bool = Field(description="Is model robust")


class DriftResponse(BaseModel):
    """Drift detection response"""
    drift_detected: bool = Field(description="Is drift detected")
    psi_score: float = Field(description="PSI score")
    ks_score: float = Field(description="KS score")
    psi_threshold: float = Field(description="PSI threshold")
    ks_threshold: float = Field(description="KS threshold")
    drift_history: List[Dict[str, float]] = Field(description="Drift history")


class PipelineRunRequest(BaseModel):
    """Request to run full pipeline"""
    config_overrides: Optional[Dict[str, Any]] = Field(None, description="Configuration overrides")
    data_file: Optional[str] = Field(None, description="Path to data file")


class PipelineRunResponse(BaseModel):
    """Response from pipeline run"""
    success: bool = Field(description="Was pipeline run successful")
    summary: Dict[str, Any] = Field(description="Run summary")
    state: Dict[str, Any] = Field(description="Final pipeline state")