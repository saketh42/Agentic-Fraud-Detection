#!/bin/bash
# Script to start both API and Streamlit services

# Start API server
echo "Starting API server..."
nohup python3 -m uvicorn app.api.server:app --host 0.0.0.0 --port 8000 > /tmp/api.log 2>&1 &

# Start Streamlit app
echo "Starting Streamlit app..."
nohup streamlit run app/frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 > /tmp/streamlit.log 2>&1 &

echo "Services started!"
echo "API: http://localhost:8000"
echo "Streamlit: http://localhost:8501"