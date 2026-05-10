#!/bin/bash
# GCP Cloud Shell Deployment Script for Agentic Fraud Detection Demo
# Run this script line by line in GCP Cloud Shell

set -e

echo "=========================================="
echo "  GCP CLOUD SHELL DEPLOYMENT"
echo "=========================================="
echo ""

# Step 1: Check environment
echo "Step 1: Checking environment..."
gcloud --version
echo ""

# Step 2: Set your project ID
echo "Step 2: Setting project ID..."
echo "Please enter your GCP Project ID (or press Enter to use existing):"
read -p "Project ID: " PROJECT_ID

if [ -z "$PROJECT_ID" ]; then
    PROJECT_ID=$(gcloud config get-value project)
    echo "Using existing project: $PROJECT_ID"
else
    gcloud config set project $PROJECT_ID
fi

echo "Project set to: $PROJECT_ID"
echo ""

# Step 3: Enable APIs
echo "Step 3: Enabling required APIs..."
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com
echo "APIs enabled!"
echo ""

# Step 4: Create Artifact Registry repository
echo "Step 4: Creating Artifact Registry repository..."
gcloud artifacts repositories create fraud-detection-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="Fraud detection demo container images" || true
echo "Artifact Registry ready!"
echo ""

# Step 5: Configure Docker
echo "Step 5: Configuring Docker authentication..."
gcloud auth configure-docker us-central1-docker.pkg.dev
echo "Docker configured!"
echo ""

# Step 6: Clone or upload your project
echo "Step 6: Project setup..."
echo "Do you need to upload your project files?"
echo "1. Yes, I'll upload files now"
echo "2. Files already exist in Cloud Shell"
read -p "Choose option (1 or 2): " UPLOAD_CHOICE

if [ "$UPLOAD_CHOICE" = "1" ]; then
    echo "Please upload your project files to Cloud Shell"
    echo "Then press Enter to continue..."
    read
fi

# Step 7: Build and deploy API
echo ""
echo "=========================================="
echo "Step 7: Building and deploying API server"
echo "=========================================="

cd /path/to/your/project  # Update this path

# Build API image
gcloud builds submit \
    --tag us-central1-docker.pkg.dev/$PROJECT_ID/fraud-detection-repo/fraud-detection-api:v1 \
    --file gcp/Dockerfile.api .

# Deploy API to Cloud Run
API_URL=$(gcloud run deploy fraud-detection-api \
    --image us-central1-docker.pkg.dev/$PROJECT_ID/fraud-detection-repo/fraud-detection-api:v1 \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8000 \
    --memory 512Mi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 10 \
    --format 'value(status.url)')

echo "API deployed at: $API_URL"
echo ""

# Step 8: Build and deploy Frontend
echo ""
echo "=========================================="
echo "Step 8: Building and deploying Streamlit UI"
echo "=========================================="

# Build Frontend image
gcloud builds submit \
    --tag us-central1-docker.pkg.dev/$PROJECT_ID/fraud-detection-repo/fraud-detection-ui:v1 \
    --file gcp/Dockerfile.frontend .

# Deploy Frontend with API URL
UI_URL=$(gcloud run deploy fraud-detection-ui \
    --image us-central1-docker.pkg.dev/$PROJECT_ID/fraud-detection-repo/fraud-detection-ui:v1 \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8501 \
    --memory 1Gi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 5 \
    --set-env-vars="API_BASE_URL=$API_URL" \
    --format 'value(status.url)')

echo "Frontend deployed at: $UI_URL"
echo ""

# Step 9: Summary
echo ""
echo "=========================================="
echo "  DEPLOYMENT COMPLETE!"
echo "=========================================="
echo ""
echo "API Server:   $API_URL"
echo "Streamlit UI: $UI_URL"
echo ""
echo "Test your deployment:"
echo "  curl $API_URL/health"
echo ""
echo "To view logs:"
echo "  gcloud run logs read fraud-detection-api --region us-central1"
echo "  gcloud run logs read fraud-detection-ui --region us-central1"
echo ""
echo "To update/redeploy:"
echo "  gcloud run deploy fraud-detection-api [options]"
echo "=========================================="