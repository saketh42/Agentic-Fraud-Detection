# Agentic Fraud Detection Demo

This demo showcases the MAPE-K (Monitor-Analyze-Plan-Execute using Knowledge) agentic fraud detection system with a modern web interface.

## Features

- **Real-time Fraud Detection**: Instant fraud prediction for single transactions or batch processing
- **Model Monitoring**: Live metrics dashboard showing model performance
- **Pipeline Status**: Visualization of the MAPE-K loop architecture
- **History Tracking**: View past pipeline runs and their performance metrics
- **Responsive UI**: Mobile-friendly interface built with Streamlit

## Architecture

The demo consists of two main components:

1. **Backend API** (FastAPI)
   - RESTful API providing access to fraud detection services
   - Built with Python FastAPI for high performance
   - Exposes endpoints for prediction, metrics, and pipeline management

2. **Frontend UI** (Streamlit)
   - Interactive web interface for users
   - Real-time visualization of model metrics
   - Intuitive forms for fraud detection
   - Comprehensive dashboard views

## Setup Instructions

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)

### Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-web.txt
   ```

2. Run the backend API:
   ```bash
   uvicorn app.api.server:app --host 0.0.0.0 --port 8000
   ```

3. In a new terminal, run the frontend UI:
   ```bash
   streamlit run app/frontend/streamlit_app.py
   ```

### Docker Deployment

1. Build and run with Docker Compose:
   ```bash
   cd app
   docker-compose up --build
   ```

2. Access the applications:
   - Frontend UI: http://localhost:8501
   - Backend API: http://localhost:8000

## API Endpoints

- `GET /api/status` - System status
- `POST /api/predict/single` - Single transaction fraud prediction
- `POST /api/predict/batch` - Batch transaction fraud prediction
- `GET /api/metrics` - Model performance metrics
- `GET /api/drift` - Drift detection status
- `GET /api/pipeline/run` - Run full pipeline

## Development

### Project Structure

```
app/
├── api/              # FastAPI backend
│   ├── routes/       # API route handlers
│   ├── models/       # Pydantic data models
│   └── server.py    # Main FastAPI application
├── frontend/         # Streamlit frontend
│   ├── pages/         # Individual page components
│   ├── utils/        # Utility functions
│   └── streamlit_app.py  # Main Streamlit application
├── Dockerfile        # Backend Docker configuration
├── Dockerfile.frontend  # Frontend Docker configuration
└── docker-compose.yml   # Docker Compose configuration
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is for demonstration purposes and showcases the capabilities of the MAPE-K agentic fraud detection system.