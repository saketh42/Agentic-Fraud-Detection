"""
Main FastAPI server for fraud detection demo
"""
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Agentic Fraud Detection API",
    description="API for the MAPE-K agentic fraud detection system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
try:
    from app.api.routes import predict, status, metrics, drift, pipeline
    app.include_router(predict.router, prefix="/api/predict", tags=["predict"])
    app.include_router(status.router, prefix="/api/status", tags=["status"])
    app.include_router(metrics.router, prefix="/api/metrics", tags=["metrics"])
    app.include_router(drift.router, prefix="/api/drift", tags=["drift"])
    app.include_router(pipeline.router, prefix="/api/pipeline", tags=["pipeline"])
except ImportError as e:
    print(f"Warning: Could not import routes: {e}")

@app.get("/")
async def root():
    return {"message": "Agentic Fraud Detection API", "status": "ok"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)