"""
FastAPI server for the MAPE-K Fraud Detection system.
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import TransactionInput, FeedbackInput
from pipeline import MAPEKPipeline
from agents import KnowledgeStore

app = FastAPI(
    title="MAPE-K Agentic Fraud Detection API",
    description="Multi-Agent MAPE-K Financial Fraud Detection and Pattern Learning System",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = MAPEKPipeline()
knowledge = KnowledgeStore()


@app.get("/")
async def root():
    return {
        "message": "MAPE-K Agentic Fraud Detection API",
        "version": "2.0.0",
        "architecture": [
            "Monitor Agent → Feature Extraction → Context Retrieval → Fraud Scoring",
            "→ LLM Reasoning → Pattern Learning → Adversarial Simulation → Planning → Execute → Knowledge Store"
        ]
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/transaction/process")
async def process_transaction(txn: TransactionInput):
    try:
        result = pipeline.process_transaction(txn.dict())
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def submit_feedback(fb: FeedbackInput):
    try:
        from agents import FeedbackAgent
        agent = FeedbackAgent()
        state = {
            'feedback_action': 'store',
            'transaction_id': fb.transaction_id,
            'feedback_type': fb.feedback_type,
            'is_correct': fb.is_correct
        }
        result = agent.run(state)
        return {
            "status": "success" if result.success else "failed",
            "data": result.data,
            "message": result.message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/transaction/{transaction_id}")
async def get_transaction(transaction_id: str):
    try:
        txn = knowledge.get_transaction(transaction_id)
        if not txn:
            raise HTTPException(status_code=404, detail="Transaction not found")
        return txn
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patterns")
async def get_patterns():
    try:
        patterns = knowledge.get_all_patterns()
        return {"patterns": patterns, "total": len(patterns)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/adversarial/{transaction_id}")
async def get_adversarial(transaction_id: str):
    try:
        variants = knowledge.get_adversarial_variants(transaction_id)
        return {
            "transaction_id": transaction_id,
            "adversarial_variants": variants,
            "total": len(variants)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics")
async def get_analytics():
    try:
        analytics = knowledge.get_analytics()
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
