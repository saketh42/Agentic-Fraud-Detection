#!/bin/bash
# Unified Demo Launcher — MAPE-K Fraud Detection System
# Starts: Flask API (with auto pipeline training) + Streamlit UI

# Fix sklearn import hang on WSL2/Windows filesystem
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo ""
echo "=========================================="
echo "  MAPE-K Agentic Fraud Detection Demo"
echo "=========================================="
echo ""

# Find the venv Python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv_new/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "ERROR: Virtual environment not found at venv_new/"
    echo "Run: python3 -m venv venv_new && $VENV_PYTHON -m pip install -r requirements.txt flask flask-cors"
    exit 1
fi

# Kill existing processes
echo "[1/4] Cleaning up existing processes..."
pkill -9 -f "simple_api" 2>/dev/null || true
pkill -9 -f "streamlit" 2>/dev/null || true
sleep 2

# Start Flask API (auto-trains model on startup)
echo "[2/4] Starting API server — pipeline will train automatically..."
cd "$SCRIPT_DIR"
nohup "$VENV_PYTHON" simple_api.py > /tmp/demo_api.log 2>&1 &
API_PID=$!
echo "  API PID: $API_PID"

# Wait for pipeline training to complete
echo "[3/4] Waiting for pipeline training..."
for i in $(seq 1 15); do
    if curl -s http://localhost:8000/health 2>/dev/null | grep -q '"ok"'; then
        echo "  API is ready!"
        break
    fi
    echo "  Waiting... ($i/15)"
    sleep 3
done

# Show training output
echo ""
echo "  --- Pipeline Training Output ---"
tail -15 /tmp/demo_api.log | sed 's/^/    /'
echo "  --------------------------------"
echo ""

# Start Streamlit
echo "[4/4] Starting Streamlit UI..."
nohup streamlit run app/frontend/streamlit_app.py --server.port 8501 --server.headless true > /tmp/demo_streamlit.log 2>&1 &
STREAMLIT_PID=$!
echo "  Streamlit PID: $STREAMLIT_PID"
sleep 3

echo ""
echo "=========================================="
echo "  DEMO READY"
echo "=========================================="
echo ""
echo "  Streamlit UI:  http://localhost:8501"
echo "  API Server:    http://localhost:8000"
echo "  Live Metrics:  http://localhost:8000/api/metrics"
echo "  Drift Status:  http://localhost:8000/api/drift"
echo "  Pipeline Info: http://localhost:8000/api/pipeline/status"
echo ""
echo "  Demo examples: cat demo_examples.txt"
echo ""
echo "  To stop:  pkill -f simple_api && pkill -f streamlit"
echo "  Logs:     tail -f /tmp/demo_api.log"
echo "            tail -f /tmp/demo_streamlit.log"
echo "=========================================="
echo ""

# Keep script running
wait
