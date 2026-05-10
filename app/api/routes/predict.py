"""
Prediction routes for the fraud detection API
"""
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import io
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.api.models import schemas
from scripts.pipeline import MAPEKPipeline
from scripts.enhanced_pipeline import EnhancedMAPEKPipeline

router = APIRouter()

# Global pipeline instance
pipeline_instance = None

def initialize_pipeline():
    """Initialize the pipeline if not already done"""
    global pipeline_instance
    if pipeline_instance is None:
        # For now, we'll create a basic pipeline
        # In a real implementation, we would load a pre-trained model
        pipeline_instance = EnhancedMAPEKPipeline()
    return pipeline_instance

@router.post("/single", response_model=schemas.PredictionResult)
async def predict_single(transaction: schemas.TransactionFeatures):
    """
    Predict fraud for a single transaction
    """
    # Initialize pipeline if needed
    pipeline = initialize_pipeline()
    
    # For demo purposes, we'll create a simple prediction
    # In a real implementation, this would use the actual model
    try:
        # Convert transaction to dict
        transaction_dict = transaction.dict()
        
        # Create a simple rule-based prediction for demo
        # This is a placeholder - in reality, we'd use the trained model
        fraud_score = 0.0
        
        # Simple rule-based scoring for demo
        # In reality, this would be replaced with actual model prediction
        urgency_score = transaction_dict.get('urgency', 0.0)
        fear_score = transaction_dict.get('fear', 0.0)
        authority_score = transaction_dict.get('authority', 0.0)
        
        # Simple heuristic for demo
        fraud_score = (urgency_score + fear_score + authority_score) / 3.0
        
        # Add some randomization for demo effect
        import random
        fraud_score = min(1.0, fraud_score + random.uniform(-0.2, 0.2))
        
        # Determine risk level
        if fraud_score >= 0.75:
            risk_level = "HIGH"
        elif fraud_score >= 0.50:
            risk_level = "MEDIUM"
        elif fraud_score >= 0.25:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        # Create response
        result = schemas.PredictionResult(
            is_fraud=fraud_score > 0.5,
            fraud_probability=fraud_score,
            risk_level=risk_level,
            model_confidence=0.85,  # Placeholder confidence
            fraud_score=fraud_score,
            label_score=0.0,  # Placeholder
            tactic_score=0.0,  # Placeholder
            feature_score=0.0,  # Placeholder
            risk_features=[],  # Placeholder
            rule_breakdown={"label": 0.0, "tactic": 0.0, "feature": 0.0},  # Placeholder
            pattern_name="UNKNOWN",  # Placeholder
            pattern_type="UNKNOWN",  # Placeholder
            pattern_confidence=0.0  # Placeholder
        )
        
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Prediction failed: {str(e)}"}
        )

@router.post("/batch", response_model=schemas.BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    """
    Predict fraud for a batch of transactions from CSV file
    """
    try:
        # Read the uploaded CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # For demo purposes, we'll generate random predictions
        # In a real implementation, this would use the actual model
        predictions = []
        
        # Generate random predictions for demo
        for index, row in df.iterrows():
            # Simple random fraud score for demo
            import random
            fraud_score = random.uniform(0, 1)
            
            # Determine risk level
            if fraud_score >= 0.75:
                risk_level = "HIGH"
            elif fraud_score >= 0.50:
                risk_level = "MEDIUM"
            elif fraud_score >= 0.25:
                risk_level = "LOW"
            else:
                risk_level = "MINIMAL"
            
            pred = schemas.PredictionResult(
                is_fraud=fraud_score > 0.5,
                fraud_probability=fraud_score,
                risk_level=risk_level,
                model_confidence=0.85,
                fraud_score=fraud_score,
                label_score=0.0,
                tactic_score=0.0,
                feature_score=0.0,
                risk_features=[],
                rule_breakdown={"label": 0.0, "tactic": 0.0, "feature": 0.0},
                pattern_name="UNKNOWN",
                pattern_type="UNKNOWN",
                pattern_confidence=0.0
            )
            predictions.append(pred)
        
        # Create summary
        total_transactions = len(predictions)
        fraud_count = sum(1 for p in predictions if p.is_fraud)
        fraud_percentage = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0
        
        summary = {
            "total_transactions": total_transactions,
            "fraud_count": fraud_count,
            "fraud_percentage": fraud_percentage,
            "high_risk_count": sum(1 for p in predictions if p.risk_level == "HIGH"),
            "medium_risk_count": sum(1 for p in predictions if p.risk_level == "MEDIUM"),
            "low_risk_count": sum(1 for p in predictions if p.risk_level == "LOW")
        }
        
        return schemas.BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Batch prediction failed: {str(e)}"}
        )