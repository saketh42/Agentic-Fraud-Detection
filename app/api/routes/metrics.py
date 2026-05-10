"""
Metrics routes for the fraud detection API
"""
from fastapi import APIRouter
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.api.models import schemas

router = APIRouter()

@router.get("/", response_model=schemas.MetricsResponse)
async def get_metrics():
    """
    Get model metrics
    """
    # For demo purposes, we'll return sample metrics
    # In a real implementation, this would get actual metrics from the model
    metrics = schemas.MetricsResponse(
        accuracy=0.95,
        precision=0.92,
        recall=0.90,
        f1=0.91,
        roc_auc=0.98,
        true_negatives=1400,
        false_positives=80,
        false_negatives=100,
        true_positives=1380,
        fpr=0.05,
        robustness_curve=[
            {"epsilon": 0.0, "f1": 0.91},
            {"epsilon": 0.01, "f1": 0.89},
            {"epsilon": 0.05, "f1": 0.85},
            {"epsilon": 0.1, "f1": 0.82},
            {"epsilon": 0.2, "f1": 0.78}
        ],
        clean_f1=0.91,
        worst_f1=0.78,
        avg_f1=0.85,
        f1_drop=0.13,
        is_robust=True
    )
    return metrics