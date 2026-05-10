#!/bin/bash

# Kill any existing processes
pkill -9 -f "simple_api" 2>/dev/null
pkill -9 -f "streamlit" 2>/dev/null
pkill -9 -f "uvicorn" 2>/dev/null
sleep 1

echo "Starting API server..."
cd /mnt/c/Users/Priti/Desktop/Agentic-Fraud-Detection
nohup python3 simple_api.py > /tmp/flask_api.log 2>&1 &
API_PID=$!
echo "API server started with PID: $API_PID"

sleep 3

echo "Starting Streamlit..."
nohup streamlit run app/frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 > /tmp/streamlit_app.log 2>&1 &
STREAMLIT_PID=$!
echo "Streamlit started with PID: $STREAMLIT_PID"

sleep 5

echo ""
echo "=========================================="
echo "SERVICES STARTED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "API Server:  http://localhost:8000"
echo "API Health:  http://localhost:8000/health"
echo "API Docs:    http://localhost:8000/docs"
echo ""
echo "Streamlit:   http://localhost:8501"
echo ""
echo "To stop services: pkill -f simple_api && pkill -f streamlit"
echo ""
echo "=========================================="
echo ""

# Wait for user to press Ctrl+C
wait