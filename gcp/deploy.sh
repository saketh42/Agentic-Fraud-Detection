#!/bin/bash
# GCP Deployment Script for Agentic Fraud Detection Demo

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME_API="fraud-detection-api"
SERVICE_NAME_UI="fraud-detection-ui"
REPO_NAME="fraud-detection-repo"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================"
echo "  GCP Deployment Script"
echo "======================================${NC}"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: Google Cloud SDK not installed${NC}"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Prompt for project ID if not set
if [ "$PROJECT_ID" = "your-project-id" ]; then
    echo -e "${YELLOW}Please enter your GCP Project ID:${NC}"
    read -p "> " PROJECT_ID
fi

# Set project
echo -e "${GREEN}Setting up GCP project...${NC}"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo -e "${GREEN}Enabling required APIs...${NC}"
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Create Artifact Registry repository
echo -e "${GREEN}Creating Artifact Registry repository...${NC}"
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION \
    --description="Fraud detection demo container images" || true

# Configure Docker authentication
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build and push API image
echo -e "${GREEN}Building and deploying API server...${NC}"
cd /mnt/c/Users/Priti/Desktop/Agentic-Fraud-Detection

gcloud builds submit \
    --tag ${REGION}-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$SERVICE_NAME_API:v1 \
    --file gcp/Dockerfile.api .

gcloud run deploy $SERVICE_NAME_API \
    --image ${REGION}-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$SERVICE_NAME_API:v1 \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --port 8000 \
    --memory 512Mi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 10 \
    --timeout 120

API_URL=$(gcloud run services describe $SERVICE_NAME_API --region $REGION --format 'value(status.url)')

# Build and push Frontend image
echo -e "${GREEN}Building and deploying Streamlit UI...${NC}"

gcloud builds submit \
    --tag ${REGION}-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$SERVICE_NAME_UI:v1 \
    --file gcp/Dockerfile.frontend .

# Deploy frontend with environment variable for API URL
gcloud run deploy $SERVICE_NAME_UI \
    --image ${REGION}-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$SERVICE_NAME_UI:v1 \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --port 8501 \
    --memory 1Gi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 5 \
    --set-env-vars="API_BASE_URL=$API_URL"

UI_URL=$(gcloud run services describe $SERVICE_NAME_UI --region $REGION --format 'value(status.url)')

# Print deployment info
echo ""
echo -e "${GREEN}======================================"
echo "  Deployment Complete!"
echo "======================================${NC}"
echo ""
echo -e "API Server: ${YELLOW}$API_URL${NC}"
echo -e "Streamlit UI: ${YELLOW}$UI_URL${NC}"
echo ""
echo -e "${GREEN}Note:${NC} It may take a few minutes for the services to be fully operational."
echo ""
echo "To view logs:"
echo "  gcloud run logs read $SERVICE_NAME_API --region $REGION"
echo "  gcloud run logs read $SERVICE_NAME_UI --region $REGION"