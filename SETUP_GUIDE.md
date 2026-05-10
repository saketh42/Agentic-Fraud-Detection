# Agentic Fraud Detection Demo - Complete Setup Guide

## Overview
This guide will help you set up and run the complete fraud detection demo on your local machine.

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Internet connection (for downloading packages)

## Step-by-Step Setup

### Step 1: Navigate to Project Directory
Open a terminal and run:
```bash
cd /mnt/c/Users/Priti/Desktop/Agentic-Fraud-Detection
```

### Step 2: Install Required Packages
Run the following command to install all necessary packages:
```bash
pip install flask streamlit requests pydantic python-multipart --break-system-packages
```

### Step 3: Start the Services
Run the complete setup script:
```bash
bash start_complete_demo.sh
```

Or manually start each service:

**Terminal 1 - Start API Server:**
```bash
python3 simple_api.py
```

**Terminal 2 - Start Streamlit UI:**
```bash
streamlit run app/frontend/streamlit_app.py --server.port 8501
```

### Step 4: Access the Demo
Open your web browser and navigate to:
- **Streamlit UI**: http://localhost:8501
- **API Server**: http://localhost:8000
- **API Health Check**: http://localhost:8000/health
- **API Documentation**: http://localhost:8000/docs

## What Each Service Does

### 1. API Server (Flask)
- Runs on port 8000
- Provides REST API endpoints for fraud detection
- Handles prediction requests
- Returns model metrics and status

### 2. Streamlit UI
- Runs on port 8501
- Provides web-based user interface
- Shows fraud detection dashboard
- Displays model metrics and pipeline status

## Testing the Demo

### Test API Health:
```bash
curl http://localhost:8000/health
```

### Test Prediction:
```bash
curl -X POST http://localhost:8000/api/predict/single \
  -H "Content-Type: application/json" \
  -d '{"fraud_probability": 0.85}'
```

### Access Web Interface:
Open browser to http://localhost:8501

## Troubleshooting

### Port Already in Use
If you get "Address already in use" error:
```bash
# Find and kill the process using the port
pkill -f "python3 simple_api.py"
pkill -f "streamlit"
```

### Services Not Starting
Check the log files:
```bash
cat /tmp/flask_api.log
cat /tmp/streamlit_app.log
```

### Package Installation Issues
```bash
pip install --upgrade pip
pip install flask streamlit requests --break-system-packages
```

## Stopping the Services

To stop all services, press Ctrl+C in each terminal window, or run:
```bash
pkill -f "simple_api"
pkill -f "streamlit"
```

## Project Structure

```
Agentic-Fraud-Detection/
├── app/
│   ├── api/
│   │   ├── routes/        # API endpoints
│   │   ├── models/       # Data models
│   │   └── server.py     # FastAPI server
│   ├── frontend/
│   │   ├── pages/        # Streamlit pages
│   │   └── streamlit_app.py  # Main Streamlit app
│   ├── Dockerfile
│   └── docker-compose.yml
├── simple_api.py         # Flask API server
├── requirements.txt      # Dependencies
└── data/                 # Sample data
```

## Next Steps

1. Test the fraud detection on the Streamlit UI
2. Upload your own data for batch processing
3. Deploy to GCP using the provided Docker configuration

## Support

If you encounter any issues:
1. Check the log files in /tmp/
2. Verify all dependencies are installed
3. Ensure ports 8000 and 8501 are available