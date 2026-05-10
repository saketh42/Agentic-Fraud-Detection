"""
Pipeline routes for the fraud detection API
"""
from fastapi import APIRouter
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.api.models import schemas

router = APIRouter()

# Global pipeline instance
pipeline_instance = None

def initialize_pipeline():
    """Initialize the pipeline if not already done"""
    global pipeline_instance
    if pipeline_instance is None:
        # For now, we'll create a basic pipeline
        # In a real implementation, we would load a pre-trained model
        from scripts.enhanced_pipeline import EnhancedMAPEKPipeline
        pipeline_instance = EnhancedMAPEKPipeline()
    return pipeline_instance

@router.get("/run", response_model=schemas.PipelineRunResponse)
async def run_pipeline():
    """
    Run the full pipeline
    """
    # For demo purposes, we'll return a sample status
    # In a real implementation, this would run the actual pipeline
    return {"message": "Pipeline completed successfully"}