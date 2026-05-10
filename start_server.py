"""
Simple script to start the API server
"""
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    import uvicorn
    from app.api.server import app
    uvicorn.run(app, host="0.0.0.0", port=8000)