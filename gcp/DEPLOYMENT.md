# GCP Deployment Configuration for Agentic Fraud Detection Demo

## Overview
This document contains the configuration and instructions for deploying the Agentic Fraud Detection Demo to Google Cloud Platform (GCP).

## Prerequisites
1. Google Cloud account with billing enabled
2. Google Cloud SDK (gcloud) installed
3. Docker installed locally
4. Project created in Google Cloud Console

## Deployment Options

### Option 1: Cloud Run (Recommended for Serverless)

#### Step 1: Set up your GCP project
```bash
# Set your project ID
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

#### Step 2: Create Artifact Registry repository
```bash
gcloud artifacts repositories create fraud-detection-repo \
    --repository-format=docker \
    --location=us-central1
```

#### Step 3: Build and Deploy

**For API Server:**
```bash
cd /mnt/c/Users/Priti/Desktop/Agentic-Fraud-Detection

# Build the Docker image
gcloud builds submit --tag us-central1-docker.pkg.dev/$PROJECT_ID/fraud-detection-repo/api:v1 .

# Deploy to Cloud Run
gcloud run deploy fraud-detection-api \
    --image us-central1-docker.pkg.dev/$PROJECT_ID/fraud-detection-repo/api:v1 \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8000
```

**For Streamlit Frontend:**
```bash
# Build the Docker image
gcloud builds submit -t us-central1-docker.pkg.dev/$PROJECT_ID/fraud-detection-repo/frontend:v1 \
    --file Dockerfile.frontend .

# Deploy to Cloud Run
gcloud run deploy fraud-detection-ui \
    --image us-central1-docker.pkg.dev/$PROJECT_ID/fraud-detection-repo/frontend:v1 \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8501
```

## Option 2: App Engine (Full Application)

Create `app.yaml` in the project root:

```yaml
runtime: python312
env: standard

instance_class: F2

handlers:
  - url: /.*
    script: auto
    secure: always

automatic_scaling:
  min_instances: 0
  max_instances: 10

env_variables:
  PORT: "8080"
  PYTHONUNBUFFERED: "1"
```

Deploy with:
```bash
gcloud app deploy app.yaml
```

## Option 3: Cloud Functions (Individual Endpoints)

For serverless API functions, deploy each endpoint separately.

## Service URLs

After deployment, your services will be available at:
- **API Server**: https://[YOUR-API-SERVICE]-[hash]-uc.a.run.app
- **Streamlit UI**: https://[YOUR-UI-SERVICE]-[hash]-uc.a.run.app

## Configuration Files

### Dockerfile (API Server)
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
ENV PORT=8000
EXPOSE 8000

# Run server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "simple_api:app"]
```

### Dockerfile (Streamlit Frontend)
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Streamlit specific dependencies
RUN pip install --no-cache-dir streamlit plotly

# Copy application
COPY . .

# Expose port
ENV PORT=8501
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app/frontend/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

## Environment Variables

For production, set these environment variables:
- `API_BASE_URL`: URL of the API server
- `DATA_PATH`: Path to data files
- `MODEL_PATH`: Path to trained models

## Monitoring and Logging

View logs:
```bash
# Cloud Run logs
gcloud run logs read fraud-detection-api --region us-central1

# App Engine logs
gcloud app logs read
```

## Cost Optimization

1. **Cloud Run**: Pay only for compute time used
2. **Set min instances to 0** for automatic scaling to zero
3. **Use regional endpoints** to reduce latency
4. **Monitor usage** in GCP Console

## Troubleshooting

### Container fails to start
- Check logs: `gcloud run logs read [SERVICE_NAME]`
- Verify environment variables are set
- Check port configuration matches Dockerfile

### Permission issues
- Ensure IAM roles include "Cloud Run Admin"
- Verify Artifact Registry access

### Build failures
- Check Dockerfile syntax
- Verify all dependencies in requirements.txt
- Test build locally first with Docker

## Security Best Practices

1. **Never commit secrets** to repository
2. **Use Secret Manager** for API keys
3. **Enable Cloud Armor** for DDoS protection
4. **Use HTTPS only** (enforced by default)
5. **Implement rate limiting** in your application

## Next Steps

1. Set up custom domain (optional)
2. Configure HTTPS with managed certificates
3. Set up monitoring dashboards
4. Implement CI/CD pipeline