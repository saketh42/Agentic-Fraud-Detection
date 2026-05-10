"""
Simple test script to check if the API works
"""
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.api.server import app

if __name__ == "__main__":
    print("API server test script")
    print("App routes:")
    for route in app.routes:
        if hasattr(route, "path"):
            print(f"  {route.path}")