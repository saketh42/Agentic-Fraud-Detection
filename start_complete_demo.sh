#!/bin/bash
# Complete Demo Setup and Run Script

echo "================================================"
echo "  AGENTIC FRAUD DETECTION DEMO SETUP"
echo "================================================"
echo ""

# Navigate to project directory
cd /mnt/c/Users/Priti/Desktop/Agentic-Fraud-Detection

# Kill any existing processes
echo "Stopping any existing services..."
pkill -9 -f "simple_api" 2>/dev/null
pkill -9 -f "streamlit" 2>/dev/null
pkill -9 -f "uvicorn" 2>/dev/null
sleep 2

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import flask" 2>/dev/null || { echo "Installing Flask..."; pip3 install flask --break-system-packages; }
python3 -c "import streamlit" 2>/dev/null || { echo "Installing Streamlit..."; pip3 install streamlit --break-system-packages; }
echo "Dependencies OK"
echo ""

# Start API server
echo "Starting API server on port 8000..."
nohup python3 simple_api.py > /tmp/flask_api.log 2>&1 &
API_PID=$!
sleep 4

# Check if API is running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ API server is running on http://localhost:8000"
else
    echo "✗ API server failed to start. Check /tmp/flask_api.log for errors"
fi

# Start Streamlit
echo "Starting Streamlit UI on port 8501..."
nohup streamlit run app/frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 > /tmp/streamlit_app.log 2>&1 &
STREAMLIT_PID=$!
sleep 5

# Check if Streamlit is running
if ps -p $STREAMLIT_PID > /dev/null 2>&1; then
    echo "✓ Streamlit UI is running on http://localhost:8501"
else
    echo "✗ Streamlit failed to start. Check /tmp/streamlit_app.log for errors"
fi

echo ""
echo "================================================"
echo "  DEMO READY!"
echo "================================================"
echo ""
echo "Open your browser and go to:"
echo ""
echo "  • Streamlit UI: http://localhost:8501"
echo "  • API Endpoint: http://localhost:8000"
echo "  • API Health:   http://localhost:8000/health"
echo "  • API Docs:     http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the services"
echo "================================================"

# Keep script running
wait