"""
Status routes for the fraud detection API
"""
from fastapi import APIRouter
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.api.models import schemas

router = APIRouter()

@router.get("/", response_model=schemas.SystemStatus)
async def get_status():
    """
    Get system status
    """
    # For demo purposes, we'll return a sample status
    # In a real implementation, this would check actual system state
    status = schemas.SystemStatus(
        status="ok",
        model_loaded=True,
        last_trained="2026-05-08T01:36:08Z",
        pipeline_ready=True,
        drift_detected=False,
        metrics={
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.90,
            "f1": 0.91,
            "roc_auc": 0.98
        }
    )
    return status