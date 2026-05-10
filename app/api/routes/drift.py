"""
Drift detection routes for the fraud detection API
"""
from fastapi import APIRouter
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.api.models import schemas

router = APIRouter()

@router.get("/", response_model=schemas.DriftResponse)
async def get_drift():
    """
    Get drift detection status
    """
    # For demo purposes, we'll return sample drift information
    # In a real implementation, this would get actual drift metrics
    drift = schemas.DriftResponse(
        drift_detected=False,
        psi_score=0.05,
        ks_score=0.02,
        psi_threshold=0.20,
        ks_threshold=0.05,
        drift_history=[
            {"psi": 0.05, "ks": 0.02, "drift_detected": False},
            {"psi": 0.08, "ks": 0.03, "drift_detected": False},
            {"psi": 0.12, "ks": 0.04, "drift_detected": False}
        ]
    )
    return drift